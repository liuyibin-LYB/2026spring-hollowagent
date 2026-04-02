"""
树洞邮件机器人

功能：
- 监听邮箱中的新邮件
- 解析邮件内容，识别查询模式
- 调用树洞 RAG Agent 处理
- 将结果以 Markdown 格式回复

"""

import imaplib
import smtplib
import email
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.header import decode_header
import time
import re
import traceback
from datetime import datetime
import sys
import os

# 添加父目录到路径以导入主项目模块
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入配置
try:
    from email_config import (
        EMAIL_ADDRESS,
        EMAIL_AUTH_CODE,
        SMTP_SERVER,
        SMTP_PORT,
        IMAP_SERVER,
        CHECK_INTERVAL,
        SUBJECT_PREFIX,
        MAX_POSTS_PER_SEARCH,
    )
except ImportError:
    print("错误: 未找到 email_config.py，请先配置邮箱信息")
    exit(1)

# 导入树洞 Agent（从上级目录）
from agent import TreeholeRAGAgent

# Agent 前缀（用于日志）
PREFIX = "[EmailBot]"


class EmailBot:
    """邮件机器人类"""
    
    def __init__(self):
        """初始化邮件机器人"""
        self.email = EMAIL_ADDRESS
        self.auth_code = EMAIL_AUTH_CODE
        self.agent = None
        
        print(f"{PREFIX} 邮件机器人启动中...")
        print(f"{PREFIX} 监听邮箱: {self.email}")
        print(f"{PREFIX} 检查间隔: {CHECK_INTERVAL}秒")
        
        # 初始化树洞 Agent（非交互模式）
        try:
            self.agent = TreeholeRAGAgent(interactive=False)
            print(f"{PREFIX} 树洞 Agent 初始化成功")
        except Exception as e:
            print(f"{PREFIX} 错误: 无法初始化树洞 Agent - {e}")
            print(f"{PREFIX} 提示: 请先在命令行运行 'python3 agent.py' 进行交互式登录并保存 cookies")
            raise
    
    def decode_subject(self, subject):
        """解码邮件主题"""
        if subject is None:
            return ""
        
        decoded = []
        for part, encoding in decode_header(subject):
            if isinstance(part, bytes):
                decoded.append(part.decode(encoding or 'utf-8', errors='ignore'))
            else:
                decoded.append(part)
        return ''.join(decoded)
    
    def parse_prompt(self, subject: str, body: str) -> dict:
        """
        解析邮件主题和正文（仅模式2）
        
        参数:
            subject: 邮件主题（用于判断模式）
            body: 邮件正文（每行一个参数）
        
        返回:
            {
                "mode": 2,
                "question": str
            }
        """
        body = body.strip()

        # 仅支持模式2：自动检索
        return {
            "mode": 2,
            "question": body,
        }
    
    def process_prompt(self, subject: str, prompt: str) -> str:
        """
        处理用户的查询请求
        
        参数:
            subject: 邮件主题
            prompt: 邮件正文
            
        返回:
            Markdown 格式的回复
        """
        try:
            # 解析查询模式
            parsed = self.parse_prompt(subject, prompt)
            mode = parsed["mode"]
            
            print(f"{PREFIX} 查询模式: {mode}")
            
            # 构建 Markdown 回复
            response = f"# 树洞查询结果\n\n"
            response += f"**查询时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
            
            if mode == 2:
                # 智能自动检索模式
                question = parsed["question"]
                response += f"**模式**: 智能自动检索\n"
                response += f"**问题**: {question}\n\n"
                response += "---\n\n"
                
                print(f"{PREFIX} 问题: {question}")
                result = self.agent.mode_auto_search(question)
                
                # 显示搜索历史
                if result.get("search_history"):
                    response += f"**搜索过程** ({result['search_count']} 次搜索):\n\n"
                    for item in result["search_history"]:
                        response += f"{item['iteration']}. `{item['keyword']}`"
                        if item.get('reason'):
                            response += f" - {item['reason']}"
                        response += "\n"
                    response += "\n"
                
                response += result["answer"]
                
                # 添加来源信息
                if result.get("sources"):
                    response += f"\n\n---\n\n**参考来源**: {result['num_sources']} 个帖子\n"
            
            return response
            
        except Exception as e:
            error_msg = f"# 查询出错\n\n"
            error_msg += f"**错误信息**: {str(e)}\n\n"
            error_msg += "```\n"
            error_msg += traceback.format_exc()
            error_msg += "```\n\n"
            error_msg += "## 使用说明\n\n"
            error_msg += "当前仅支持模式2（自动检索），请按如下格式发送：\n\n"
            error_msg += "```\n主题: 树洞 自动检索\n正文:\n我想了解计算机图形学这门课\n```\n\n"
            
            print(f"{PREFIX} 错误: {e}")
            traceback.print_exc()
            return error_msg
    
    def send_reply(self, to: str, subject: str, body: str):
        """
        发送回复邮件
        
        参数:
            to: 收件人地址
            subject: 邮件主题
            body: 邮件正文（Markdown 格式）
        """
        try:
            msg = MIMEMultipart("alternative")
            msg["Subject"] = subject
            msg["From"] = self.email
            msg["To"] = to
            
            # 纯文本版本
            text_part = MIMEText(body, "plain", "utf-8")
            msg.attach(text_part)
            
            # HTML 版本（将 Markdown 转为 HTML）
            try:
                import markdown
                html_body = markdown.markdown(
                    body,
                    extensions=['fenced_code', 'tables', 'nl2br']
                )
                # 添加样式
                html_body = f"""
                <html>
                <head>
                    <style>
                        body {{ font-family: Arial, sans-serif; line-height: 1.6; }}
                        code {{ background: #f4f4f4; padding: 2px 5px; border-radius: 3px; }}
                        pre {{ background: #f4f4f4; padding: 10px; border-radius: 5px; overflow-x: auto; }}
                        blockquote {{ border-left: 3px solid #ccc; margin-left: 0; padding-left: 20px; }}
                    </style>
                </head>
                <body>
                    {html_body}
                </body>
                </html>
                """
                html_part = MIMEText(html_body, "html", "utf-8")
                msg.attach(html_part)
            except ImportError:
                pass  # 如果没有 markdown 库，只发送纯文本
            
            with smtplib.SMTP_SSL(SMTP_SERVER, SMTP_PORT) as server:
                server.login(self.email, self.auth_code)
                server.sendmail(self.email, to, msg.as_string())
            
            print(f"{PREFIX} 已发送回复到 {to}")
            
        except Exception as e:
            print(f"{PREFIX} 发送邮件失败: {e}")
            traceback.print_exc()
    
    def check_inbox(self):
        """检查收件箱中的未读邮件"""
        try:
            # 连接到 IMAP 服务器
            mail = imaplib.IMAP4_SSL(IMAP_SERVER)
            mail.login(self.email, self.auth_code)
            mail.select("inbox")
            
            # 搜索未读邮件
            status, messages = mail.search(None, "UNSEEN")
            if status != "OK":
                mail.logout()
                return
            
            message_ids = messages[0].split()
            
            if message_ids:
                print(f"{PREFIX} 发现 {len(message_ids)} 封未读邮件")
            
            for num in message_ids:
                try:
                    # 获取邮件内容
                    status, data = mail.fetch(num, "(RFC822)")
                    if status != "OK":
                        continue
                    
                    msg = email.message_from_bytes(data[0][1])
                    from_addr = msg.get("From")
                    subject = self.decode_subject(msg.get("Subject", ""))
                    
                    # 提取发件人邮箱地址
                    if '<' in from_addr:
                        from_email = re.search(r'<(.+?)>', from_addr).group(1)
                    else:
                        from_email = from_addr
                    
                    print(f"{PREFIX} 处理邮件: {subject} (来自 {from_email})")
                    
                    # 检查主题关键词
                    if SUBJECT_PREFIX not in subject:
                        print(f"{PREFIX} 跳过: 主题不符合要求（需要包含 '{SUBJECT_PREFIX}'）")
                        # 标记为已读
                        mail.store(num, "+FLAGS", "\\Seen")
                        continue
                    
                    # 提取邮件正文
                    body = ""
                    if msg.is_multipart():
                        for part in msg.walk():
                            content_type = part.get_content_type()
                            if content_type == "text/plain":
                                try:
                                    payload = part.get_payload(decode=True)
                                    charset = part.get_content_charset() or 'utf-8'
                                    body = payload.decode(charset, errors='ignore')
                                    break
                                except Exception as e:
                                    print(f"{PREFIX} 解析邮件正文失败: {e}")
                    else:
                        try:
                            payload = msg.get_payload(decode=True)
                            charset = msg.get_content_charset() or 'utf-8'
                            body = payload.decode(charset, errors='ignore')
                        except Exception as e:
                            print(f"{PREFIX} 解析邮件正文失败: {e}")
                    
                    if not body.strip():
                        print(f"{PREFIX} 跳过: 邮件正文为空")
                        mail.store(num, "+FLAGS", "\\Seen")
                        continue
                    
                    # 标记为已读（避免重复处理）
                    mail.store(num, "+FLAGS", "\\Seen")
                    
                    # 处理查询
                    print(f"{PREFIX} 开始处理查询...")
                    reply_body = self.process_prompt(subject, body)
                    
                    # 生成不含"树洞"的回信标题，避免触发回环
                    parsed = self.parse_prompt(subject, body)
                    mode = parsed["mode"]
                    if mode == 2:
                        q = parsed.get('question', '').replace('\n', ' ')[:25]
                        reply_subject = f"[检索结果] {q}"
                    else:
                        reply_subject = "[检索结果] 树洞查询"
                    self.send_reply(from_email, reply_subject, reply_body)
                    
                    print(f"{PREFIX} 查询处理完成")
                    
                except Exception as e:
                    print(f"{PREFIX} 处理邮件时出错: {e}")
                    traceback.print_exc()
            
            mail.logout()
            
        except Exception as e:
            print(f"{PREFIX} 检查邮箱时出错: {e}")
            traceback.print_exc()
    
    def run(self):
        """运行邮件机器人（主循环）"""
        print(f"{PREFIX} 邮件机器人已启动，开始监听...")
        print(f"{PREFIX} 按 Ctrl+C 停止")
        
        while True:
            try:
                self.check_inbox()
                time.sleep(CHECK_INTERVAL)
            except KeyboardInterrupt:
                print(f"\n{PREFIX} 收到停止信号，正在退出...")
                break
            except Exception as e:
                print(f"{PREFIX} 运行时错误: {e}")
                traceback.print_exc()
                time.sleep(CHECK_INTERVAL)


def main():
    """主函数"""
    try:
        bot = EmailBot()
        bot.run()
    except Exception as e:
        print(f"{PREFIX} 启动失败: {e}")
        traceback.print_exc()
        exit(1)


if __name__ == "__main__":
    main()
