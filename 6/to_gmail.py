import numpy
import pandas

import smtplib
from email.mime.text import MIMEText
from email.header import Header
from email.utils import formatdate

FROM_ADDRESS = "asdaad@server"
TO_ADDRESS = [
    #aaaaaaaaaaaaaaaa@gmail.com
    ]


def send(msg):
    #msg = make_message()
    send_email(msg)

def make_message():
    return "test"


def send_email(message):
    charset = "ISO-2022-JP"
    subject = u"予測結果v6"
    msg = MIMEText(message.encode(charset), "plain", charset)
    msg["Subject"] = Header(subject, charset)
    msg["From"]    = FROM_ADDRESS
    msg["To"]      = ",".join(TO_ADDRESS)
    msg["Date"]    = formatdate(localtime=True)


    smtp = smtplib.SMTP("localhost")
    smtp.sendmail(FROM_ADDRESS, TO_ADDRESS, msg.as_string())
    print("mail send")
    smtp.close()