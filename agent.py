import asyncio
import re
from enum import Enum
from dataclasses import dataclass
from typing import Any, List, Optional, Dict

from transformers import pipeline

# --------------------------
# Enums and Data Structures
# --------------------------

class TicketCategory(Enum):
    TECHNICAL = "technical"
    BILLING = "billing"
    FEATURE = "feature"
    ACCESS = "access"

class Priority(Enum):
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    URGENT = 4

@dataclass
class TicketAnalysis:
    category: TicketCategory
    priority: Priority
    key_points: List[str]
    required_expertise: List[str]
    sentiment: float
    urgency_indicators: List[str]
    business_impact: str
    suggested_response_type: str

@dataclass
class ResponseSuggestion:
    response_text: str
    confidence_score: float
    requires_approval: bool
    suggested_actions: List[str]

@dataclass
class SupportTicket:
    id: str
    subject: str
    content: str
    customer_info: Dict[str, Any]

@dataclass
class TicketResolution:
    analysis: TicketAnalysis
    response: ResponseSuggestion

# --------------------------
# TicketAnalysisAgent
# --------------------------

class TicketAnalysisAgent:
    def __init__(self):
        # Pipeline for zero-shot classification to determine ticket category.
        self.classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
        # Pipeline for sentiment analysis.
        self.sentiment_analyzer = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
        # Pipeline for summarization to extract key points.
        self.summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")

    async def analyze_ticket(
        self,
        ticket_content: str,
        customer_info: Optional[dict] = None
    ) -> TicketAnalysis:
        content_lower = ticket_content.lower()
        
        # ----- 1. Ticket Classification using Zero-Shot Learning -----
        candidate_labels = ["technical issue", "billing question", "feature request", "account access"]
        classification_result = self.classifier(ticket_content, candidate_labels=candidate_labels)
        selected_label = classification_result["labels"][0].lower()
        
        # Map selected label to our TicketCategory enum.
        if "billing" in selected_label:
            category = TicketCategory.BILLING
        elif "feature" in selected_label:
            category = TicketCategory.FEATURE
        elif "access" in selected_label:
            category = TicketCategory.ACCESS
        else:
            category = TicketCategory.TECHNICAL

        # ----- 2. Sentiment Analysis -----
        sentiment_result = self.sentiment_analyzer(ticket_content)
        sentiment_label = sentiment_result[0]["label"]
        sentiment = -0.5 if sentiment_label == "NEGATIVE" else 0.5

        # ----- 3. Priority Assessment -----
        urgency_words = ["asap", "urgent", "immediately", "cannot", "crash", "error"]
        detected_urgency = [word for word in urgency_words if word in content_lower]

        role = customer_info.get("role", "").lower() if customer_info else ""
        high_priority_roles = ["director", "admin", "c-level", "chief", "executive"]
        impact_keywords = ["payroll", "demo", "revenue", "critical"]

        if detected_urgency or any(kw in content_lower for kw in impact_keywords) or any(hr in role for hr in high_priority_roles):
            priority = Priority.URGENT
        else:
            priority = Priority.MEDIUM

        # ----- 4. Key Point Extraction via Summarization -----
        # Generate a summary (this might be refined to extract bullet points in production)
        summary_output = self.summarizer(ticket_content, max_length=50, min_length=25, do_sample=False)[0]["summary_text"]
        # Split summary into individual sentences or key points.
        key_points = [kp.strip() for kp in re.split(r'\.|\n', summary_output) if kp.strip()]

        # ----- 5. Expertise Mapping & Business Impact -----
        expertise_mapping = {
            TicketCategory.ACCESS: ["System Admin"],
            TicketCategory.BILLING: ["Billing Specialist"],
            TicketCategory.TECHNICAL: ["Technical Support"],
            TicketCategory.FEATURE: ["Product Manager"]
        }
        required_expertise = expertise_mapping.get(category, [])
        business_impact = "High" if any(kw in content_lower for kw in impact_keywords) else "Low"
        suggested_response_type = "immediate" if priority == Priority.URGENT else "standard"

        return TicketAnalysis(
            category=category,
            priority=priority,
            key_points=key_points,
            required_expertise=required_expertise,
            sentiment=sentiment,
            urgency_indicators=detected_urgency,
            business_impact=business_impact,
            suggested_response_type=suggested_response_type
        )

# --------------------------
# ResponseAgent
# --------------------------

class ResponseAgent:
    def __init__(self):
        # Text generation pipeline for dynamic, personalized responses.
        self.generator = pipeline("text-generation", model="gpt2")

    async def generate_response(
        self,
        ticket_analysis: TicketAnalysis,
        response_templates: Dict[str, str],
        context: Dict[str, Any]
    ) -> ResponseSuggestion:
        # Build a prompt using ticket analysis and context.
        prompt = (
            f"Generate a personalized customer support response. "
            f"Ticket Category: {ticket_analysis.category.value}. "
            f"Priority: {ticket_analysis.priority.name}. "
            f"Sentiment Score: {ticket_analysis.sentiment}. "
            f"Business Impact: {ticket_analysis.business_impact}. "
            f"Key Points: {', '.join(ticket_analysis.key_points)}. "
            f"Required Expertise: {', '.join(ticket_analysis.required_expertise)}. "
            f"Customer Name: {context.get('customer_name', 'Customer')}. "
            "Include a greeting, explanation of the issue, proposed resolution steps, and a friendly closing."
        )
        
        # Generate the response text.
        generated = self.generator(prompt, max_length=150, num_return_sequences=1)
        response_text = generated[0]["generated_text"]

        # Optionally, you could further post-process the generated text to align with a template.
        confidence_score = 0.90  # Placeholder for demonstration.
        requires_approval = ticket_analysis.priority == Priority.URGENT
        suggested_actions = (
            ["Follow up with customer", "Escalate to senior support"]
            if requires_approval
            else ["Monitor ticket"]
        )

        return ResponseSuggestion(
            response_text=response_text,
            confidence_score=confidence_score,
            requires_approval=requires_approval,
            suggested_actions=suggested_actions
        )

# --------------------------
# TicketProcessor Orchestration
# --------------------------

class TicketProcessor:
    def __init__(self):
        self.analysis_agent = TicketAnalysisAgent()
        self.response_agent = ResponseAgent()
        self.context = {}

    async def process_ticket(
        self,
        ticket: SupportTicket,
    ) -> TicketResolution:
        try:
            # Update context from customer info.
            self.context["customer_name"] = ticket.customer_info.get("name", "Customer")
            # Analyze the ticket.
            analysis = await self.analysis_agent.analyze_ticket(ticket.content, ticket.customer_info)
            # Define (or load) response templates (fallback if needed).
            response_templates = {
                "access_issue": (
                    "Hello {name},\n\n"
                    "I understand you're having trouble accessing the {feature}. Let me help you resolve this.\n\n"
                    "{diagnosis}\n\n"
                    "{resolution_steps}\n\n"
                    "Priority Status: {priority_level}\n"
                    "Estimated Resolution: {eta}\n\n"
                    "Please let me know if you need any clarification.\n\n"
                    "Best regards,\n"
                    "Support Team"
                ),
                "billing_inquiry": (
                    "Hi {name},\n\n"
                    "Thank you for your inquiry about {billing_topic}.\n\n"
                    "{explanation}\n\n"
                    "{next_steps}\n\n"
                    "If you have any questions, don't hesitate to ask.\n\n"
                    "Best regards,\n"
                    "Billing Team"
                )
            }
            # Generate a response using the LLM-powered ResponseAgent.
            response = await self.response_agent.generate_response(analysis, response_templates, self.context)
            return TicketResolution(analysis=analysis, response=response)
        except Exception as e:
            print(f"Error processing ticket {ticket.id}: {str(e)}")
            raise e

# --------------------------
# Example Test Harness
# --------------------------

async def main():
    # Sample tickets from test data.
    sample_tickets = [
        SupportTicket(
            id="TKT-001",
            subject="Cannot access admin dashboard",
            content=(
                "Hi Support,\n"
                "Since this morning I can't access the admin dashboard. I keep getting a 403 error.\n"
                "I need this fixed ASAP as I need to process payroll today.\n\n"
                "Thanks,\n"
                "John Smith\n"
                "Finance Director"
            ),
            customer_info={"role": "Admin", "plan": "Enterprise", "name": "John Smith"}
        ),
        SupportTicket(
            id="TKT-002",
            subject="Question about billing cycle",
            content=(
                "Hello,\n"
                "Our invoice shows billing from the 15th but we signed up on the 20th.\n"
                "Can you explain how the pro-rating works?\n\n"
                "Best regards,\n"
                "Sarah Jones"
            ),
            customer_info={"role": "Billing Admin", "plan": "Professional", "name": "Sarah Jones"}
        )
    ]

    processor = TicketProcessor()
    for ticket in sample_tickets:
        resolution = await processor.process_ticket(ticket)
        print(f"Ticket ID: {ticket.id}")
        print("---- Analysis ----")
        print(resolution.analysis)
        print("\n---- Response Suggestion ----")
        print(resolution.response)
        print("\n" + "-" * 50 + "\n")

if __name__ == "__main__":
    asyncio.run(main())
