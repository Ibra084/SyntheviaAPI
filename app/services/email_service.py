from email.message import EmailMessage
import aiosmtplib
from app.core.config import settings

async def send_email(msg: EmailMessage):
    await aiosmtplib.send(
        msg,
        hostname="smtp.gmail.com",
        port=587,
        username=settings.GMAIL_USER,
        password=settings.GMAIL_PASS,
        start_tls=True,
    )
