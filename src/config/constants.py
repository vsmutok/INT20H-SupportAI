SEED = 42

# Intent mapping from Bitext categories to our categories
# 5 core intents from the task + "other" as fallback
INTENT_MAP = {
    # payment_issue
    "payment_issue": "payment_issue",
    "check_payment_methods": "payment_issue",
    "check_invoice": "payment_issue",
    "get_invoice": "payment_issue",
    # technical_error
    "registration_problems": "technical_error",
    # account_access
    "edit_account": "account_access",
    "delete_account": "account_access",
    "recover_password": "account_access",
    "switch_account": "account_access",
    "create_account": "account_access",
    # tariff_question
    "check_cancellation_fee": "tariff_question",
    "newsletter_subscription": "tariff_question",
    # refund_request
    "get_refund": "refund_request",
    "track_refund": "refund_request",
    "check_refund_policy": "refund_request",
    "cancel_order": "refund_request",
    # other (everything else)
    "change_shipping_address": "other",
    "set_up_shipping_address": "other",
    "change_order": "other",
    "track_order": "other",
    "place_order": "other",
    "delivery_options": "other",
    "delivery_period": "other",
    "complaint": "other",
    "review": "other",
    "contact_human_agent": "other",
    "contact_customer_service": "other",
}

# Valid intent labels (for analyzer validation)
VALID_INTENTS = ["payment_issue", "technical_error", "account_access", "tariff_question", "refund_request", "other"]

AGENT_MISTAKES = [
    "ignored_question",
    "incorrect_info",
    "rude_tone",
    "no_resolution",
    "unnecessary_escalation",
]

HIDDEN_DISSATISFACTION_CLOSINGS = [
    "Thanks for your response",
    "Okay, I understand",
    "Fine, I'll try that",
    "Alright then",
    "Got it, thanks anyway",
    "Sure, whatever",
    "Okay...",
    "Well, I guess that's all I can do",
    "I see. Thanks for trying",
    "Right, I'll figure it out myself then",
]

# Template variable replacements — realistic placeholder values
TEMPLATE_REPLACEMENTS = {
    "{{Order Number}}": ["ORD-48291", "ORD-73650", "ORD-12784", "ORD-55903", "ORD-31267"],
    "{{Account Type}}": ["Premium", "Gold", "Business", "Professional", "Enterprise"],
    "{{Account Category}}": ["Standard", "Basic", "Silver", "Starter", "Personal"],
    "{{Person Name}}": ["John Smith", "Sarah Johnson", "Michael Chen", "Emily Davis", "Alex Martinez"],
    "{{Invoice Number}}": ["INV-20240315", "INV-20240287", "INV-20240401", "INV-20240198", "INV-20240522"],
    "{{Refund Amount}}": ["$49.99", "$125.00", "$79.50", "$199.99", "$34.95"],
    "{{Tracking Number}}": ["TRK-8827364", "TRK-5519082", "TRK-3304817", "TRK-9912045", "TRK-6678293"],
    "{{Customer Support Phone Number}}": ["1-800-555-0199", "1-888-555-0147", "1-877-555-0163"],
    "{{Customer Support Hours}}": ["Monday-Friday, 9 AM - 6 PM EST", "24/7", "Mon-Sat, 8 AM - 8 PM"],
    "{{Website URL}}": ["www.example.com", "support.example.com", "help.example.com"],
    "{{Currency Symbol}}": ["$", "$", "$"],
    "{{Delivery City}}": ["New York", "Los Angeles", "Chicago", "Houston", "Seattle"],
    "{{Delivery Country}}": ["United States", "Canada", "United Kingdom"],
    "{{Client Last Name}}": ["Smith", "Johnson", "Williams", "Brown", "Davis"],
    "{{Salutation}}": ["Dear Customer", "Hello", "Hi there"],
    "{{Online Order Interaction}}": ["online portal", "website order page", "mobile app"],
    "{{Online Company Portal Info}}": ["customer portal at portal.example.com", "your account dashboard"],
    "{{Settings}}": ["Account Settings", "Settings", "Preferences"],
    "{{Profile}}": ["My Profile", "Account Profile", "User Profile"],
    "{{Profile Type}}": ["Premium Profile", "Business Profile", "Professional Profile"],
    "{{Store Location}}": ["our downtown store", "the Main Street branch", "Store #42"],
    "{{Login Page URL}}": ["login.example.com", "www.example.com/login"],
    "{{Product Name}}": ["CloudSync Pro", "DataGuard Plus", "StreamLine Basic"],
    "{{Subscription Type}}": ["Monthly", "Annual", "Premium Monthly"],
    "{{App}}": ["our mobile app", "the desktop application", "our web platform"],
    "{{Company}}": ["TechCorp", "CloudNet Solutions", "Digital Services Inc."],
    "{{Promo Code}}": ["SAVE20", "WELCOME15", "LOYALTY10"],
    "{{Pricing Plan}}": ["Basic Plan ($9.99/mo)", "Pro Plan ($24.99/mo)", "Enterprise Plan ($49.99/mo)"],
}
