from tools_2 import update_cold_email, _load_cold_emails
import json

print("Testing update_cold_email with recipient_name='Dr. Davies'...")
result = update_cold_email(recipient_name="Dr. Davies", status="responded", notes="Test response from script")
print(f"Result: {result}")

# Verify file content
data = _load_cold_emails()
for email in data["emails"]:
    if email["recipient_name"] == "Dr. Davies":
        print(f"Verified in file: Status={email['status']}, Notes={email['notes']}")
