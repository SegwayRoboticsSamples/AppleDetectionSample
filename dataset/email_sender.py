import smtplib
from email.mime.text import MIMEText
from email.header import Header


def send_email(email_subject, email_text):
    from_addr = "peng.chen@ninebot.com"
    password = "ZXCAsdpass9bot12"
    to_addr = "rui.li@ninebot.com"
    smtp_server = "smtp.qiye.163.com"

    msg = MIMEText(email_text, 'plain', 'utf-8')
    msg['From'] = from_addr
    msg['To'] = to_addr
    msg['Subject'] = Header(email_subject, 'utf-8').encode()

    server = smtplib.SMTP()
    server.connect(smtp_server)
    server.login(from_addr, password)
    server.sendmail(from_addr, [to_addr], msg.as_string())
    server.quit()


# send_email("test python smtp", "test email.")