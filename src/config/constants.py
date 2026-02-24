SEED = 42

# Intent mapping from Bitext categories to our categories
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
    "change_shipping_address": "account_access",

    # tariff_question 
    "check_cancellation_fee": "tariff_question",

    # refund_request 
    "get_refund": "refund_request",
    "track_refund": "refund_request",
    "check_refund_policy": "refund_request",
    "cancel_order": "refund_request",
    "complaint": "other",
    "delivery_problem": "other",
    "track_order": "other",
    "place_order": "other",
    "delivery_options": "other",
    "contact_human_agent": "other",
    "contact_customer_service": "other",
    "newsletter_subscription": "other",
}

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
]
