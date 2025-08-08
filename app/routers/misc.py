from fastapi import APIRouter, BackgroundTasks
from email.message import EmailMessage
from app.models.schemas import ContactForm
from app.core.config import settings
from app.services.email_service import send_email

router = APIRouter(tags=["Misc"])

@router.get("/health")
def health():
    return {"status": "ok"}

@router.post("/contact")
async def contact(form: ContactForm, bg: BackgroundTasks):
    msg = EmailMessage()
    msg["From"] = settings.GMAIL_USER
    msg["To"] = "ibrahim.rahman@synthevia.com"
    msg["Reply-To"] = form.email
    msg["Subject"] = f"[Contact] {form.subject}"
    msg.set_content(
        f"Name: {form.name}\nEmail: {form.email}\nSubject: {form.subject}\n\n{form.message}"
    )
    msg.add_alternative(f"""
    <html><body>
      <h2>New Contact Form Submission</h2>
      <p><strong>Name:</strong> {form.name}</p>
      <p><strong>Email:</strong> {form.email}</p>
      <p><strong>Subject:</strong> {form.subject}</p>
      <pre>{form.message}</pre>
    </body></html>
    """, subtype="html")
    bg.add_task(send_email, msg)
    return {"message": "Email queued"}
