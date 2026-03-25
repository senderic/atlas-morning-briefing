
import pytest
from unittest.mock import MagicMock, patch
from scripts.email_distributor import EmailDistributor

def test_distribute_multiple_recipients():
    # Mock SMTP
    with patch("scripts.email_distributor.smtplib.SMTP") as mock_smtp:
        mock_server = MagicMock()
        mock_smtp.return_value.__enter__.return_value = mock_server
        
        distributor = EmailDistributor(
            sender_email="sender@gmail.com",
            sender_password="password"
        )
        
        config = {
            "email_recipients": ["a@b.com, c@d.com", "e@f.com", "a@b.com"]
        }
        
        results = distributor.distribute(
            config=config,
            markdown_content="Hello world",
            subject="Test Subject"
        )
        
        # Should have sent to 3 addresses (a@b.com is deduplicated)
        assert len(results) == 3
        assert "a@b.com" in results
        assert "c@d.com" in results
        assert "e@f.com" in results
        
        # Verify server.send_message called ONCE with all recipients in To field
        assert mock_server.send_message.call_count == 1
        args, kwargs = mock_server.send_message.call_args
        msg = args[0]
        to_field = msg["To"]
        assert "a@b.com" in to_field
        assert "c@d.com" in to_field
        assert "e@f.com" in to_field
