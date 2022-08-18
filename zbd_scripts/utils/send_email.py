import time
import os
import smtplib
from email.mime.text import MIMEText
from email.header import Header


# 即使不用来发文件，记录控制台输出也是极好的
# 使用时，用add_log替换print函数，最后在os.system('shutdown')前调用send_mail函数。
class Mylog:
    def __init__(self, header, subject):
        self.subject = subject
        self.header = header

        time_info = str(time.strftime("%m%d_%H%M%S", time.localtime()))
        self.time_info = time_info

        if not os.path.exists('./email_logs/'):
            os.mkdir('./email_logs/')
        self.file_name = './email_logs/' + time_info + '.log'
        self.file = open(self.file_name, 'a')

    def __del__(self):
        self.file.close()

    # 添加日志记录
    def add_log(self, lg):
        time_info = str(time.strftime("%H:%M:%S -->  ", time.localtime()))
        self.file.write(time_info + lg)
        self.file.write('\r')
        self.file.flush()

    # 添加日志记录并在终端输出
    def print_add(self, lg):
        print(lg)
        self.add_log(lg)

    # 运行结束后发送邮件
    def send_mail(self):
        self.file.close()
        from_addr = '15939437879@163.com'  # 邮件发送账号
        to_addrs = '379837872@qq.com'  # 接收邮件账号
        qqCode = 'GVCAMHOVGWGDWSXK'  # 授权码（这个要填自己获取到的）
        smtp_server = 'smtp.163.com'
        smtp_port = 465
        # 配置服务器
        stmp = smtplib.SMTP_SSL(smtp_server, smtp_port)
        stmp.login(from_addr, qqCode)
        with open(self.file_name, 'r') as f:
            buffer = f.read()
        # 组装发送内容
        message = MIMEText(buffer, 'plain', 'utf-8')  # 发送的内容
        message['From'] = Header(self.header, 'utf-8')  # 发件人
        message['Subject'] = Header(self.subject + ', start_time:' + self.time_info, 'utf-8')  # 邮件标题
        try:
            stmp.sendmail(from_addr, to_addrs, message.as_string())
            print('邮件发送成功')
        except Exception as e:
            print('邮件发送失败--' + str(e))