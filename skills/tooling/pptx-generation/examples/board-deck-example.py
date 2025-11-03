#!/usr/bin/env python3
"""
Board Deck Example - Executive-Level Strategic Presentation
Demonstrates complete board presentation with financial data, strategic initiatives,
and executive summary using constraint-based design principles.

Features:
- Multi-section structure with clear narrative flow
- Financial data visualization
- Strategic initiative tracking
- Risk assessment matrix
- Accessibility compliance (WCAG AA)
- Clean, professional design without decorative elements
"""

import sys
import os
from pathlib import Path

# Add resources directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'resources'))

from slide_generator import (
    SlideGenerator,
    ColorPalette,
    LayoutConstraints,
    SlideType
)


def create_board_deck(output_path: str = "board-deck.html"):
    """
    Generate comprehensive board deck presentation

    Structure:
    1. Title Slide
    2. Executive Summary Section
    3. Financial Performance
    4. Strategic Initiatives
    5. Risk Assessment
    6. Recommendations
    """

    # Configure professional color palette
    # Deep blue conveys trust and stability for financial presentations
    palette = ColorPalette(
        primary="#1E3A8A",      # Deep blue - primary brand
        secondary="#3B82F6",    # Medium blue - secondary accent
        accent="#F59E0B",       # Amber - highlights and emphasis
        background="#FFFFFF",   # White - clean, professional
        text="#1F2937"          # Dark gray - readable text
    )

    # Standard layout constraints for board presentations
    constraints = LayoutConstraints(
        min_margin_inches=0.5,
        min_font_size_body=18,      # Readable from distance
        min_font_size_title=24,     # Clear hierarchy
        max_bullets_per_slide=3,    # Force conciseness
        min_line_spacing=1.2        # Comfortable reading
    )

    # Initialize generator
    generator = SlideGenerator(palette, constraints)

    print("Generating Board Deck Presentation...")
    print("=" * 60)

    # ========================================================================
    # SLIDE 1: Title Slide
    # ========================================================================
    print("Creating slide 1: Title")
    generator.create_title_slide(
        title="Strategic Initiative Review",
        subtitle="Board of Directors Meeting | Q4 2024"
    )

    # ========================================================================
    # SECTION 1: Executive Summary
    # ========================================================================
    print("Creating section: Executive Summary")
    generator.create_section_break("Executive Summary")

    generator.create_content_slide(
        title="Financial Highlights",
        bullets=[
            "Revenue reached $127.3M (+23% YoY), exceeding targets by $8.2M",
            "Operating margin improved to 32.1% (Q3: 28.4%), driven by automation",
            "Cash position strengthened to $42.8M, supporting growth investments"
        ],
        accent_bar=True
    )

    generator.create_content_slide(
        title="Strategic Progress",
        bullets=[
            "Enterprise product launch ahead of schedule, 47 pilot customers secured",
            "International expansion initiated in EMEA with 3 strategic partnerships",
            "Platform modernization 78% complete, on track for Q1 2025 release"
        ],
        accent_bar=True
    )

    generator.create_content_slide(
        title="Key Challenges",
        bullets=[
            "Market headwinds in SMB segment require pricing strategy adjustment",
            "Talent acquisition slower than planned (32 vs 45 target hires)",
            "Supply chain constraints impacting delivery timelines by 2-3 weeks"
        ],
        accent_bar=True
    )

    # ========================================================================
    # SECTION 2: Financial Performance
    # ========================================================================
    print("Creating section: Financial Performance")
    generator.create_section_break("Financial Performance")

    generator.create_two_column_slide(
        title="Revenue Growth Analysis",
        left_content="""Q4 2024 Performance:

• Total Revenue: $127.3M
• Recurring Revenue: $98.7M (77.5%)
• New Business: $28.6M
• Expansion Revenue: $14.2M

Year-over-Year Growth:
• Overall: +23.1%
• Enterprise: +34.2%
• Mid-Market: +18.7%
• SMB: +8.3%""",
        right_content="""Revenue by Segment:

Enterprise (45%): $57.3M
- Strong adoption of premium tier
- Average contract value: $420K
- Retention rate: 96%

Mid-Market (35%): $44.6M
- Steady growth trajectory
- Focus on vertical solutions
- Retention rate: 91%

SMB (20%): $25.4M
- Pricing pressure evident
- Strategic reassessment needed
- Retention rate: 84%"""
    )

    generator.create_content_slide(
        title="Operating Margin Improvement",
        bullets=[
            "Gross margin: 78.2% (Q3: 76.1%) through automation and scale",
            "Operating expenses: $58.4M, 45.9% of revenue (Q3: 48.2%)",
            "Path to 35% operating margin by Q3 2025 remains achievable"
        ],
        accent_bar=True
    )

    generator.create_two_column_slide(
        title="Cash Flow & Balance Sheet",
        left_content="""Cash Flow Statement:

Operating Cash Flow: $38.2M
- Collections strong at 94%
- Days sales outstanding: 42

Free Cash Flow: $29.7M
- CapEx: $8.5M (infrastructure)
- Growth investments on track""",
        right_content="""Balance Sheet Strength:

Cash & Equivalents: $42.8M
Total Assets: $214.6M
Total Liabilities: $67.3M
Stockholders' Equity: $147.3M

Debt-to-Equity: 0.18
Current Ratio: 2.7
Quick Ratio: 2.3"""
    )

    # ========================================================================
    # SECTION 3: Strategic Initiatives
    # ========================================================================
    print("Creating section: Strategic Initiatives")
    generator.create_section_break("Strategic Initiatives")

    generator.create_content_slide(
        title="Enterprise Product Launch",
        bullets=[
            "Launched October 15 with 47 pilot customers (target: 35-40)",
            "Early feedback: 8.7/10 satisfaction, 94% would recommend",
            "Revenue pipeline: $12.4M in Q1 2025, $31M annual run-rate projected"
        ],
        accent_bar=True
    )

    generator.create_two_column_slide(
        title="International Expansion - EMEA",
        left_content="""Market Entry Strategy:

Phase 1 (Q4 2024): COMPLETED
- Strategic partner agreements: 3
- Regulatory compliance: Complete
- Localization: UK, Germany, France

Phase 2 (Q1 2025): IN PROGRESS
- Direct sales team: 8 hires
- Channel partnerships: Target 12
- Marketing campaigns: Launch Feb""",
        right_content="""Early Performance Indicators:

Pilot Deployments: 14 customers
- UK: 8 customers
- Germany: 4 customers
- France: 2 customers

Revenue Booked: €2.3M
Average Deal Size: €164K
Sales Cycle: 87 days (vs 62 US)

2025 Target: €15M revenue"""
    )

    generator.create_content_slide(
        title="Platform Modernization",
        bullets=[
            "78% complete: Core services migrated, UI refresh 85% done",
            "Performance improvements: 40% faster load times, 62% lower latency",
            "Q1 2025 launch on track, beta program with 120 customers"
        ],
        accent_bar=True
    )

    generator.create_two_column_slide(
        title="AI/ML Capabilities Integration",
        left_content="""Development Progress:

Smart Recommendations:
- Accuracy: 87% (target: 85%)
- Adoption: 64% of users
- Engagement lift: +28%

Predictive Analytics:
- Models deployed: 4 of 6
- Forecasting accuracy: 91%
- Customer interest: High""",
        right_content="""Business Impact Forecast:

Revenue Opportunity:
- Upsell attach rate: +15%
- Premium tier conversion: +22%
- Estimated impact: $8.2M ARR

Cost Efficiency:
- Support ticket reduction: -18%
- Automation hours saved: 12K
- Operating cost savings: $2.4M"""
    )

    # ========================================================================
    # SECTION 4: Risk Assessment
    # ========================================================================
    print("Creating section: Risk Assessment")
    generator.create_section_break("Risk Assessment")

    generator.create_content_slide(
        title="Market & Competitive Risks",
        bullets=[
            "SMB pricing pressure from low-cost competitors requires response",
            "Enterprise sales cycles lengthening (87 vs 76 days avg)",
            "Talent war intensifying: 23% salary inflation in engineering roles"
        ],
        accent_bar=True
    )

    generator.create_two_column_slide(
        title="Operational Risk Matrix",
        left_content="""HIGH PRIORITY RISKS:

Supply Chain Delays:
- Impact: Delivery timelines +2-3wks
- Probability: High (ongoing)
- Mitigation: Dual sourcing, buffer

Talent Acquisition Gaps:
- Impact: Product roadmap delays
- Probability: Medium
- Mitigation: Contractor bridge""",
        right_content="""MEDIUM PRIORITY RISKS:

Technology Infrastructure:
- Impact: Scaling challenges
- Probability: Low
- Mitigation: Platform upgrade

Regulatory Changes (EMEA):
- Impact: Compliance costs
- Probability: Medium
- Mitigation: Legal monitoring"""
    )

    generator.create_content_slide(
        title="Financial Risks & Hedging",
        bullets=[
            "Currency exposure: €8.2M unhedged EMEA revenue (monitoring)",
            "Customer concentration: Top 5 = 34% revenue (diversifying)",
            "Market downturn scenario: 18-month runway at current burn"
        ],
        accent_bar=True
    )

    # ========================================================================
    # SECTION 5: Recommendations & Next Steps
    # ========================================================================
    print("Creating section: Recommendations")
    generator.create_section_break("Recommendations")

    generator.create_content_slide(
        title="Board Recommendations",
        bullets=[
            "Approve $15M growth investment for EMEA expansion and enterprise sales",
            "Authorize pricing strategy revision for SMB segment (launch Q1 2025)",
            "Greenlight executive talent search for Chief Revenue Officer role"
        ],
        accent_bar=True
    )

    generator.create_two_column_slide(
        title="Q1 2025 Priorities",
        left_content="""Revenue Growth:
- Enterprise: Scale to 60+ customers
- EMEA: €4M in bookings
- Platform launch: 250+ migrations

Operational Excellence:
- Hiring: Close 28 open positions
- Efficiency: Reduce CAC by 12%
- Retention: Maintain 92%+ NRR""",
        right_content="""Strategic Initiatives:
- Product: Complete AI/ML rollout
- Partnerships: Sign 8 channel deals
- M&A: Evaluate 3 targets

Financial Targets:
- Revenue: $145M (+14% QoQ)
- Operating margin: 33.5%
- Free cash flow: $32M"""
    )

    generator.create_content_slide(
        title="Key Milestones - Next 90 Days",
        bullets=[
            "Jan 15: SMB pricing strategy approved, Feb 1 implementation",
            "Feb 10: EMEA sales team onboarding complete, quota attainment begins",
            "Mar 20: Platform modernization GA release, customer migration complete"
        ],
        accent_bar=True
    )

    # ========================================================================
    # Export Presentation
    # ========================================================================
    print("\nExporting presentation...")
    result = generator.export_for_html2pptx(output_path)

    print("=" * 60)
    print("Board Deck Generation Complete!")
    print("=" * 60)
    print(f"Output file: {result['output_path']}")
    print(f"Total slides: {result['slide_count']}")
    print(f"\nValidation Summary:")
    print(f"  Contrast validation: {'PASS' if result['validation_summary']['all_contrast_valid'] else 'FAIL'}")
    print(f"  Font size validation: {'PASS' if result['validation_summary']['all_fonts_valid'] else 'FAIL'}")
    print(f"  Prohibited elements: {result['validation_summary']['prohibited_violations']}")

    if result['validation_summary']['prohibited_violations'] == 0:
        print("\n✓ Presentation meets all design constraints")
    else:
        print(f"\n⚠ Warning: {result['validation_summary']['prohibited_violations']} constraint violations detected")

    print(f"\nNext step: Convert to PPTX")
    print(f"  Command: python -m html2pptx {output_path} board-deck.pptx")

    return result


if __name__ == "__main__":
    # Generate board deck
    create_board_deck()
