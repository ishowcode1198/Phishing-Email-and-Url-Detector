import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


def build_dataset():
    phishing_examples = [
        "Your account has been suspended. Verify your login details immediately.",
        "Click here to update your payment information to avoid service interruption.",
        "Urgent: your bank account will be closed unless you verify your identity now.",
        "Dear customer, we noticed unusual activity. Confirm your password here.",
        "You have won a prize! Claim the reward by entering your credentials.",
        "Sign in to your account using the link below to prevent termination.",
        "Update billing information for your PayPal account now.",
        "Security alert: unauthorized login attempt detected. Reset your password.",
        "Confirm your email address to restore full account access.",
        "Verify your identity by clicking the secure verification link.",
        "Please verify your account here: http://secure-login.example.com/verify.",
        "Urgent security notice: update your billing details at https://account-security.example.com/login.",
        "Your mailbox is over quota. Click here to verify now.",
        "Payment failed. Update your billing info at http://account-update.example.com.",
        "We detected unusual sign-in activity. Validate your account immediately.",
        "The package delivery failed. Confirm your shipping details at www.shipping-verify.example.com.",
        "Your Netflix account has been locked. Reactivate at https://secure-verify.example.com.",
        "Important: verify your Apple ID using the following link.",
        "Your bank account will be closed unless you confirm your password.",
        "Reset your password at http://login.example.com/reset to restore access.",
        "You are eligible for a reward. Provide your credentials to claim it.",
        "Account suspension warning: sign in to avoid service interruption.",
        "Click this link to receive your payment: https://claim-reward.example.com.",
        "Confirm your identity with the secure login portal below.",
    ]

    safe_examples = [
        "Hi team, please find the monthly report attached.",
        "Can you send me the updated project plan by Thursday?",
        "Meeting rescheduled to 3 PM tomorrow. Let me know if that works.",
        "Your order has been shipped and will arrive within three business days.",
        "Reminder: submit your timesheet before the end of the week.",
        "This is a friendly follow-up about the marketing proposal.",
        "Please review the attached invoice and confirm if everything is correct.",
        "The event location has changed, and the new address is included below.",
        "Your subscription renewal is complete. Thank you for staying with us.",
        "Welcome to the platform! We have created your account successfully.",
        "Visit our blog for the latest updates: https://example.com/blog.",
        "The conference agenda is available at www.company-events.com/agenda.",
        "Our support portal is available at https://support.example.com for more help.",
        "The conference agenda is available at https://events.example.com/agenda.",
        "Your profile has been updated successfully.",
        "Thank you for your purchase. Track your shipment online.",
        "We received your request and will process it shortly.",
        "Team, the lunch meeting is in conference room two.",
        "Use two-factor authentication to secure your account.",
        "Your invoice is attached; no action is required unless there is an issue.",
        "Visit our homepage at https://example.com to learn more.",
        "The project plan is in the shared drive; all team members can access it.",
        "Please review the attached document and share feedback.",
        "Join the monthly strategy call at 10 AM on Tuesday.",
    ]

    texts = phishing_examples + safe_examples
    labels = ["not_safe"] * len(phishing_examples) + ["safe"] * len(safe_examples)
    return texts, labels


def train_and_save_model(model_path="model.pkl"):
    texts, labels = build_dataset()
    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.25, random_state=42, stratify=labels
    )

    pipeline = make_pipeline(
        TfidfVectorizer(ngram_range=(1, 2), stop_words="english", min_df=1, max_df=0.9),
        LogisticRegression(max_iter=5000, class_weight="balanced", random_state=42),
    )

    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_test)

    print("Model evaluation:\n")
    print(classification_report(y_test, predictions, target_names=["not_safe", "safe"]))

    with open(model_path, "wb") as model_file:
        pickle.dump(pipeline, model_file)

    print(f"Saved trained model to {model_path}")


if __name__ == "__main__":
    train_and_save_model()
