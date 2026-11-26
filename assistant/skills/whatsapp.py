"""WhatsApp Web automation with Selenium.

Design notes
  * Selenium is synchronous and not thread-safe → ALL driver work runs on one dedicated
    worker thread, so the asyncio loop (mic, STT, TTS) never blocks.
  * A persistent Chrome profile keeps you logged in; scan the QR code once.
  * Messages are only sent to contacts in config.yaml, and ONLY after the user says
    "yes" to a read-back (see PendingAction). Free-form speech never picks a recipient.
  * WhatsApp's DOM changes often. If sending breaks, update the two selectors below.
  * Automating WhatsApp Web may violate WhatsApp's terms; keep it to low-volume personal
    use, or move to the official WhatsApp Business Cloud API for anything serious.
"""
from __future__ import annotations

import asyncio
import difflib
import logging
import re
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import quote

from ..config import WhatsAppCfg
from ..core.context import ConversationContext
from ..core.skill import Skill
from ..core.types import Intent, PendingAction, SkillResult
from ..errors import SkillError
from ..utils.text import normalize

log = logging.getLogger(__name__)

_PHONE = re.compile(r"\+?\d{8,15}")
MAX_MESSAGE_CHARS = 1000


class WhatsAppWebClient:
    COMPOSER_XPATH = '//footer//div[@contenteditable="true"]'
    SEND_XPATH = '//footer//button[@aria-label="Send"] | //footer//span[@data-icon="send"]'

    def __init__(self, cfg: WhatsAppCfg) -> None:
        self._cfg = cfg
        self._driver = None
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="whatsapp")

    # -- runs on the worker thread --------------------------------------
    def _ensure_driver(self):
        if self._driver is not None:
            try:
                _ = self._driver.title  # raises if the browser window was closed
                return self._driver
            except Exception:  # noqa: BLE001
                self._reset_driver()
        try:
            from selenium import webdriver
        except ImportError as exc:
            raise SkillError("selenium missing", user_message="Selenium isn't installed.") from exc
        opts = webdriver.ChromeOptions()
        opts.add_argument(f"--user-data-dir={Path(self._cfg.profile_dir).resolve()}")
        opts.add_argument("--no-first-run")
        if self._cfg.headless:
            opts.add_argument("--headless=new")
        try:
            self._driver = webdriver.Chrome(options=opts)
        except Exception as exc:  # noqa: BLE001
            raise SkillError(str(exc), user_message="I couldn't start Chrome for WhatsApp.") from exc
        return self._driver

    def _reset_driver(self) -> None:
        driver, self._driver = self._driver, None
        if driver is not None:
            try:
                driver.quit()
            except Exception:  # noqa: BLE001
                pass

    def _send_sync(self, phone: str, text: str) -> None:
        try:
            from selenium.common.exceptions import TimeoutException, WebDriverException
            from selenium.webdriver.common.by import By
            from selenium.webdriver.support import expected_conditions as EC
            from selenium.webdriver.support.ui import WebDriverWait
        except ImportError as exc:
            raise SkillError("selenium missing", user_message="Selenium isn't installed.") from exc

        driver = self._ensure_driver()
        try:
            driver.get(f"https://web.whatsapp.com/send?phone={phone.lstrip('+')}&text={quote(text)}")
            # First run: this waits while you scan the QR code.
            WebDriverWait(driver, self._cfg.login_timeout_s).until(
                EC.presence_of_element_located((By.XPATH, self.COMPOSER_XPATH))
            )
            button = WebDriverWait(driver, self._cfg.send_timeout_s).until(
                EC.element_to_be_clickable((By.XPATH, self.SEND_XPATH))
            )
            button.click()
            time.sleep(1.5)  # let the outgoing message flush before any navigation
        except TimeoutException as exc:
            raise SkillError(
                "whatsapp web timed out",
                user_message="WhatsApp Web didn't load. Make sure you're logged in.",
            ) from exc
        except WebDriverException as exc:
            self._reset_driver()
            raise SkillError(str(exc), user_message="Something went wrong with WhatsApp Web.") from exc

    # -- async API ------------------------------------------------------
    async def send(self, phone: str, text: str) -> None:
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(self._executor, self._send_sync, phone, text)

    async def aclose(self) -> None:
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(self._executor, self._reset_driver)
        self._executor.shutdown(wait=False)


class WhatsAppSkill(Skill):
    name = "whatsapp"
    description = "Sends a WhatsApp message to a saved contact."
    slot_hints = '{"contact": "<name>", "message": "<text to send>"}'
    priority = 50
    patterns = (
        r"\bsend\s+(?:a\s+)?(?:whatsapp\s+)?(?:message|msg|text)\s+to\s+(?P<contact>.+?)\s+"
        r"(?:saying|that\s+says|that|to\s+say)\s+(?P<message>.+)$",
        r"\btell\s+(?P<contact>.+?)\s+(?:on|in)\s+whatsapp\s+(?:that\s+)?(?P<message>.+)$",
        r"\bwhatsapp\s+(?P<contact>\w+)\s+(?:saying|that)\s+(?P<message>.+)$",
    )

    def __init__(self, cfg: WhatsAppCfg, client: WhatsAppWebClient | None = None) -> None:
        super().__init__()
        self._client = client or WhatsAppWebClient(cfg)
        self._contacts: dict[str, str] = {}
        for name, phone in cfg.contacts.items():
            phone = str(phone).replace(" ", "").replace("-", "")
            if _PHONE.fullmatch(phone):
                self._contacts[normalize(name)] = phone
            else:
                log.warning("ignoring WhatsApp contact %r: bad phone number", name)

    def _lookup(self, spoken: str) -> str | None:
        key = normalize(spoken)
        if key in self._contacts:
            return self._contacts[key]
        close = difflib.get_close_matches(key, list(self._contacts), n=1, cutoff=0.8)
        return self._contacts[close[0]] if close else None

    async def handle(self, intent: Intent, ctx: ConversationContext) -> SkillResult:
        contact = intent.slots.get("contact", "").strip()
        message = intent.slots.get("message", "").strip()
        if not contact or not message:
            return SkillResult("Who should I message, and what should I say?", ok=False)
        phone = self._lookup(contact)
        if phone is None:
            return SkillResult(f"I don't have {contact} in your WhatsApp contacts.", ok=False)
        message = message[:MAX_MESSAGE_CHARS]

        async def _send() -> SkillResult:
            await self._client.send(phone, message)
            return SkillResult(f"Sent to {contact}.")

        return SkillResult(
            speech=f"I'll send {contact} this message: {message}. Should I send it?",
            pending=PendingAction(
                execute=_send,
                cancel_speech="Okay, I won't send it.",
                description=f"whatsapp:{contact}",
            ),
        )

    async def aclose(self) -> None:
        await self._client.aclose()
