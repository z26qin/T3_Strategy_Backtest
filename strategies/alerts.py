"""
Automated Alerts Module

Supports Email and Discord notifications for trading signals.
"""

import smtplib
import requests
import json
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from dataclasses import dataclass
from typing import Optional, Dict
from datetime import datetime


@dataclass
class AlertConfig:
    """Alert configuration."""
    # Email settings
    email_enabled: bool = False
    smtp_server: str = "smtp.gmail.com"
    smtp_port: int = 587
    email_sender: str = ""
    email_password: str = ""  # Use App Password for Gmail
    email_recipient: str = "z26qin@uwaterloo.ca"

    # Discord settings
    discord_enabled: bool = False
    discord_webhook_url: str = ""


class AlertManager:
    """Manage and send trading alerts."""

    def __init__(self, config: Optional[AlertConfig] = None):
        self.config = config or AlertConfig()

    def send_alert(self, signal: str, summary: Dict) -> Dict[str, bool]:
        """Send alert through all enabled channels."""
        results = {}

        message = self._format_message(signal, summary)

        if self.config.email_enabled:
            results['email'] = self._send_email(signal, message)

        if self.config.discord_enabled:
            results['discord'] = self._send_discord(signal, summary)

        return results

    def _format_message(self, signal: str, summary: Dict) -> str:
        """Format alert message."""
        message = f"""
TQQQ STRATEGY ALERT

Signal: {signal}
Date: {summary.get('date', 'N/A')}
Position: {summary.get('position', 'N/A')}

MARKET DATA:
- QQQ Close: ${summary.get('qqq_close', 0):.2f}
- QQQ Daily Change: {summary.get('qqq_daily_return', 0):+.2f}%
- TQQQ Close: ${summary.get('tqqq_close', 0):.2f}

LEVELS:
- MA200: ${summary.get('ma200', 0):.2f}
- Buy Level: ${summary.get('buy_level', 0):.2f}
- Sell Level: ${summary.get('sell_level', 0):.2f}

CONDITIONS:
- QQQ > Buy Level: {'YES' if summary.get('conditions', {}).get('above_buy_level') else 'NO'}
- Daily Loss >= 1%: {'YES' if summary.get('conditions', {}).get('daily_loss_met') else 'NO'}
- QQQ < Sell Level: {'YES' if summary.get('conditions', {}).get('below_sell_level') else 'NO'}
"""
        return message.strip()

    def _send_email(self, signal: str, message: str) -> bool:
        """Send email alert."""
        try:
            msg = MIMEMultipart()
            msg['From'] = self.config.email_sender
            msg['To'] = self.config.email_recipient
            msg['Subject'] = f"TQQQ Alert: {signal} Signal - {datetime.now().strftime('%Y-%m-%d')}"

            msg.attach(MIMEText(message, 'plain'))

            with smtplib.SMTP(self.config.smtp_server, self.config.smtp_port) as server:
                server.starttls()
                server.login(self.config.email_sender, self.config.email_password)
                server.send_message(msg)

            print(f"Email alert sent successfully to {self.config.email_recipient}")
            return True
        except Exception as e:
            print(f"Email alert failed: {e}")
            return False

    def _send_discord(self, signal: str, summary: Dict) -> bool:
        """Send Discord alert via webhook."""
        try:
            color = {"BUY": 0x00FF00, "SELL": 0xFF0000, "HOLD": 0x808080}.get(signal, 0x808080)

            embed = {
                "title": f"TQQQ Strategy Alert: {signal}",
                "color": color,
                "fields": [
                    {"name": "Date", "value": summary.get('date', 'N/A'), "inline": True},
                    {"name": "Position", "value": summary.get('position', 'N/A'), "inline": True},
                    {"name": "QQQ Close", "value": f"${summary.get('qqq_close', 0):.2f}", "inline": True},
                    {"name": "QQQ Daily Change", "value": f"{summary.get('qqq_daily_return', 0):+.2f}%", "inline": True},
                    {"name": "TQQQ Close", "value": f"${summary.get('tqqq_close', 0):.2f}", "inline": True},
                    {"name": "MA200", "value": f"${summary.get('ma200', 0):.2f}", "inline": True},
                    {"name": "Buy Level", "value": f"${summary.get('buy_level', 0):.2f}", "inline": True},
                    {"name": "Sell Level", "value": f"${summary.get('sell_level', 0):.2f}", "inline": True},
                ],
                "timestamp": datetime.utcnow().isoformat()
            }

            payload = {"embeds": [embed]}
            response = requests.post(
                self.config.discord_webhook_url,
                json=payload,
                timeout=10
            )

            if response.status_code in [200, 204]:
                print("Discord alert sent successfully")
                return True
            else:
                print(f"Discord alert failed: {response.text}")
                return False
        except Exception as e:
            print(f"Discord alert failed: {e}")
            return False

    def test_alerts(self) -> Dict[str, bool]:
        """Test all enabled alert channels."""
        test_summary = {
            'date': datetime.now().strftime('%Y-%m-%d'),
            'signal': 'TEST',
            'position': 'TEST MODE',
            'qqq_close': 500.00,
            'qqq_daily_return': -1.5,
            'tqqq_close': 75.00,
            'ma200': 480.00,
            'buy_level': 499.20,
            'sell_level': 465.60,
            'conditions': {
                'above_buy_level': True,
                'daily_loss_met': True,
                'below_sell_level': False,
            }
        }
        return self.send_alert("TEST", test_summary)


def load_config_from_env() -> AlertConfig:
    """Load alert configuration from environment variables."""
    import os

    return AlertConfig(
        email_enabled=os.getenv('ALERT_EMAIL_ENABLED', '').lower() == 'true',
        smtp_server=os.getenv('ALERT_SMTP_SERVER', 'smtp.gmail.com'),
        smtp_port=int(os.getenv('ALERT_SMTP_PORT', '587')),
        email_sender=os.getenv('ALERT_EMAIL_SENDER', ''),
        email_password=os.getenv('ALERT_EMAIL_PASSWORD', ''),
        email_recipient=os.getenv('ALERT_EMAIL_RECIPIENT', 'z26qin@uwaterloo.ca'),
        discord_enabled=os.getenv('ALERT_DISCORD_ENABLED', '').lower() == 'true',
        discord_webhook_url=os.getenv('ALERT_DISCORD_WEBHOOK_URL', ''),
    )


def load_config_from_file(filepath: str) -> AlertConfig:
    """Load alert configuration from JSON file."""
    with open(filepath, 'r') as f:
        data = json.load(f)
    return AlertConfig(**data)


# Example usage and setup instructions
SETUP_INSTRUCTIONS = """
=== ALERT SETUP INSTRUCTIONS ===

1. EMAIL (Gmail):
   - Enable 2FA on your Google account
   - Generate App Password: Google Account > Security > App Passwords
   - Set environment variables:
     export ALERT_EMAIL_ENABLED=true
     export ALERT_EMAIL_SENDER=your.email@gmail.com
     export ALERT_EMAIL_PASSWORD=your_app_password
     export ALERT_EMAIL_RECIPIENT=z26qin@uwaterloo.ca

2. DISCORD:
   - In your Discord server: Server Settings > Integrations > Webhooks
   - Create webhook and copy URL
   - Set environment variables:
     export ALERT_DISCORD_ENABLED=true
     export ALERT_DISCORD_WEBHOOK_URL=your_webhook_url

Or create a config.json file:
{
    "email_enabled": true,
    "smtp_server": "smtp.gmail.com",
    "smtp_port": 587,
    "email_sender": "your.email@gmail.com",
    "email_password": "your_app_password",
    "email_recipient": "z26qin@uwaterloo.ca",
    "discord_enabled": true,
    "discord_webhook_url": "your_webhook_url"
}
"""


if __name__ == "__main__":
    print(SETUP_INSTRUCTIONS)

    # Example with environment variables
    config = load_config_from_env()
    alert_manager = AlertManager(config)

    # Test alerts
    results = alert_manager.test_alerts()
    print(f"\nTest Results: {results}")
