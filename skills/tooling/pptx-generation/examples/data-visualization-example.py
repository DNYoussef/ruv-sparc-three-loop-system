#!/usr/bin/env python3
"""
Data Visualization Example - Analytics Dashboard Presentation
Demonstrates data-driven presentation with charts, metrics, and insights
using integrated chart generation and clean design.

Features:
- Integrated matplotlib chart generation
- Financial and operational metrics
- Trend analysis and forecasting
- Multi-dimensional data views
- Clean, uncluttered visual design
"""

import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'resources'))

from slide_generator import (
    SlideGenerator,
    ColorPalette,
    LayoutConstraints
)
from chart_creator import ChartCreator, ChartConfig


def create_data_visualization_presentation(output_path: str = "data-visualization.html"):
    """
    Generate comprehensive data visualization presentation

    Demonstrates:
    - Chart integration with slides
    - Multiple chart types
    - Data storytelling
    - Insights and recommendations
    """

    # Color palette optimized for data visualization
    palette = ColorPalette(
        primary="#1E3A8A",      # Deep blue
        secondary="#3B82F6",    # Medium blue
        accent="#F59E0B",       # Amber for highlights
        background="#FFFFFF",   # White background
        text="#1F2937"          # Dark gray text
    )

    constraints = LayoutConstraints(
        min_margin_inches=0.5,
        min_font_size_body=18,
        min_font_size_title=24,
        max_bullets_per_slide=3,
        min_line_spacing=1.2
    )

    # Initialize slide generator
    generator = SlideGenerator(palette, constraints)

    # Initialize chart creator
    chart_config = ChartConfig(
        color_palette=['#1E3A8A', '#3B82F6', '#60A5FA', '#93C5FD', '#DBEAFE'],
        background_color='#FFFFFF',
        text_color='#1F2937',
        title_font_size=20,
        label_font_size=14,
        tick_font_size=12
    )
    chart_creator = ChartCreator(chart_config)

    print("Generating Data Visualization Presentation...")
    print("=" * 70)

    # ========================================================================
    # TITLE
    # ========================================================================
    print("Creating slides: Title & Overview")

    generator.create_title_slide(
        title="Q4 2024 Analytics Review",
        subtitle="Performance Metrics & Growth Analysis"
    )

    generator.create_section_break("Executive Summary")

    # ========================================================================
    # KEY METRICS OVERVIEW
    # ========================================================================
    generator.create_two_column_slide(
        title="Key Performance Indicators",
        left_content="""Revenue Metrics:

Total Revenue: $127.3M
• MRR: $10.6M (+18% YoY)
• ARR: $127.2M (+22% YoY)
• Average Deal Size: $42K

Growth Rates:
• Revenue Growth: +23% YoY
• Customer Growth: +34%
• User Growth: +41%""",
        right_content="""Operational Metrics:

Customer Metrics:
• Active Customers: 3,142
• New Customers (Q4): 287
• Churn Rate: 2.3%
• NRR: 118%

Efficiency:
• CAC: $8,420 (-12% vs Q3)
• LTV: $156,000
• LTV:CAC Ratio: 18.5:1
• Payback Period: 8 months"""
    )

    # ========================================================================
    # REVENUE ANALYSIS
    # ========================================================================
    print("Creating section: Revenue Analysis")
    generator.create_section_break("Revenue Analysis")

    # Generate revenue growth chart
    print("  Generating revenue chart...")
    revenue_data = {
        'Q1': 98.2,
        'Q2': 107.5,
        'Q3': 115.8,
        'Q4': 127.3
    }

    revenue_chart = chart_creator.create_bar_chart(
        revenue_data,
        'Quarterly Revenue Trend',
        x_label='Quarter',
        y_label='Revenue ($M)'
    )

    # Save chart
    chart_path = Path(output_path).parent / "charts"
    chart_path.mkdir(exist_ok=True)
    revenue_chart_file = chart_path / "revenue_chart.png"
    chart_creator.save_chart(revenue_chart, str(revenue_chart_file))

    # Chart reference slide (in actual implementation, embed the chart)
    generator.create_two_column_slide(
        title="Revenue Growth Trajectory",
        left_content="""Quarterly Performance:

Q1 2024: $98.2M
• YoY Growth: +19%
• QoQ Growth: +8%

Q2 2024: $107.5M
• YoY Growth: +21%
• QoQ Growth: +9%

Q3 2024: $115.8M
• YoY Growth: +22%
• QoQ Growth: +8%

Q4 2024: $127.3M
• YoY Growth: +23%
• QoQ Growth: +10%""",
        right_content="""Growth Drivers:

Enterprise Segment:
• +$18.2M incremental
• 34% growth rate
• 12 new enterprise deals

Product Expansion:
• +$8.4M upsells
• 94% customer adoption
• 3 new product launches

Geographic:
• +$4.7M international
• EMEA expansion
• APAC pilot program"""
    )

    generator.create_content_slide(
        title="Revenue Composition Analysis",
        bullets=[
            "Recurring revenue: 77.5% ($98.7M), up from 72% in Q3",
            "Professional services: 15.2% ($19.3M), margin improvement to 42%",
            "One-time licenses: 7.3% ($9.3M), strategic phase-out continuing"
        ],
        accent_bar=True
    )

    # ========================================================================
    # CUSTOMER ANALYTICS
    # ========================================================================
    print("Creating section: Customer Analytics")
    generator.create_section_break("Customer Analytics")

    # Generate customer growth chart
    print("  Generating customer growth chart...")
    customer_series = {
        'Total Customers': [
            (1, 2547), (2, 2683), (3, 2891), (4, 3142)
        ],
        'Enterprise': [
            (1, 342), (2, 378), (3, 421), (4, 487)
        ]
    }

    customer_chart = chart_creator.create_line_chart(
        customer_series,
        'Customer Growth by Segment',
        x_label='Quarter',
        y_label='Customer Count'
    )

    customer_chart_file = chart_path / "customer_chart.png"
    chart_creator.save_chart(customer_chart, str(customer_chart_file))

    generator.create_two_column_slide(
        title="Customer Acquisition & Retention",
        left_content="""Acquisition Metrics:

New Customers (Q4): 287
• Enterprise: 66 (+41%)
• Mid-Market: 128 (+35%)
• SMB: 93 (+18%)

Acquisition Channels:
• Direct Sales: 142 (49%)
• Channel Partners: 89 (31%)
• Self-Service: 56 (20%)

CAC by Channel:
• Direct: $12,400
• Partners: $6,200
• Self-Service: $1,800""",
        right_content="""Retention Performance:

Churn Analysis:
• Overall Churn: 2.3%
• Enterprise: 1.1%
• Mid-Market: 2.8%
• SMB: 4.7%

Retention Initiatives:
• CSM program: -0.8% churn
• Onboarding improve: -0.5%
• Product engagement: +12%

Net Revenue Retention:
• Q4 NRR: 118%
• Expansion: $14.2M
• Contraction: $3.1M"""
    )

    generator.create_content_slide(
        title="Customer Lifetime Value Analysis",
        bullets=[
            "Average LTV increased to $156K (+8% vs Q3) driven by lower churn",
            "LTV:CAC ratio improved to 18.5:1, exceeding 15:1 target",
            "Payback period reduced to 8 months from 9 months in Q3"
        ],
        accent_bar=True
    )

    # ========================================================================
    # PRODUCT USAGE ANALYTICS
    # ========================================================================
    print("Creating section: Product Analytics")
    generator.create_section_break("Product Usage Analytics")

    # Generate feature adoption pie chart
    print("  Generating adoption chart...")
    adoption_data = {
        'Core Features': 94.2,
        'Advanced Analytics': 67.8,
        'AI Capabilities': 52.3,
        'API Integration': 38.5
    }

    adoption_chart = chart_creator.create_pie_chart(
        adoption_data,
        'Feature Adoption Rates',
        show_percentages=True
    )

    adoption_chart_file = chart_path / "adoption_chart.png"
    chart_creator.save_chart(adoption_chart, str(adoption_chart_file))

    generator.create_two_column_slide(
        title="Product Engagement Metrics",
        left_content="""Daily Active Users:

Total DAU: 47,320
• Week-over-week: +3.2%
• Month-over-month: +12%

DAU/MAU Ratio: 0.68
• Industry benchmark: 0.4
• Indicates high stickiness

Session Metrics:
• Avg sessions/day: 2.8
• Avg session length: 23min
• Actions per session: 18""",
        right_content="""Feature Usage Insights:

Most Used Features:
1. Dashboard (94% adoption)
2. Reports (89% adoption)
3. Analytics (68% adoption)
4. AI Insights (52% adoption)

Power User Analysis:
• Top 20%: 78% of activity
• Daily users: 67% of base
• Mobile users: 34% (+8%)

Usage Patterns:
• Peak: 10am-2pm EST
• Mobile: Evening/weekend
• API: Automated/continuous"""
    )

    generator.create_content_slide(
        title="Product Performance Metrics",
        bullets=[
            "Page load time: 1.2s average (target: <2s), 15% improvement",
            "API response time: P95 at 340ms, meeting 500ms SLA",
            "Uptime: 99.97% (SLA: 99.9%), only 13 minutes downtime"
        ],
        accent_bar=True
    )

    # ========================================================================
    # SALES & MARKETING ANALYTICS
    # ========================================================================
    print("Creating section: Sales & Marketing")
    generator.create_section_break("Sales & Marketing Performance")

    # Generate pipeline funnel
    print("  Generating funnel visualization...")
    funnel_data = {
        'Leads': 12400,
        'MQLs': 3720,
        'SQLs': 1240,
        'Opportunities': 496,
        'Closed-Won': 142
    }

    # Create horizontal bar for funnel visualization
    funnel_chart = chart_creator.create_bar_chart(
        funnel_data,
        'Sales Funnel Conversion',
        x_label='Count',
        y_label='Stage',
        horizontal=True
    )

    funnel_chart_file = chart_path / "funnel_chart.png"
    chart_creator.save_chart(funnel_chart, str(funnel_chart_file))

    generator.create_two_column_slide(
        title="Sales Pipeline Analysis",
        left_content="""Pipeline Metrics:

Total Pipeline: $42.8M
• Weighted: $18.3M
• Stage 3+: $24.6M
• Closing Q1: $12.4M

Conversion Rates:
• Lead → MQL: 30%
• MQL → SQL: 33%
• SQL → Opp: 40%
• Opp → Close: 29%

Avg Deal Size:
• Enterprise: $142K
• Mid-Market: $38K
• SMB: $12K""",
        right_content="""Sales Cycle Metrics:

Average Sales Cycle:
• Enterprise: 87 days
• Mid-Market: 42 days
• SMB: 18 days

Win Rate Analysis:
• Overall: 29%
• Competitive: 24%
• Uncontested: 41%

Lost Deal Reasons:
1. Price (42%)
2. Features (28%)
3. Timing (18%)
4. Competition (12%)"""
    )

    generator.create_content_slide(
        title="Marketing Campaign Performance",
        bullets=[
            "Content marketing: 4,200 MQLs generated, $1,850 CAC",
            "Paid search: 1,800 MQLs, $3,200 CAC, ROAS 4.2:1",
            "Partner referrals: 920 MQLs, $980 CAC, highest close rate 38%"
        ],
        accent_bar=True
    )

    # ========================================================================
    # FINANCIAL ANALYSIS
    # ========================================================================
    print("Creating section: Financial Analysis")
    generator.create_section_break("Financial Performance")

    # Generate profitability scatter plot
    print("  Generating profitability analysis...")
    segment_profitability = [
        (142, 45),  # Enterprise: avg deal size, margin %
        (38, 32),   # Mid-market
        (12, 18)    # SMB
    ]

    profit_chart = chart_creator.create_scatter_plot(
        segment_profitability,
        'Segment Profitability Analysis',
        x_label='Average Deal Size ($K)',
        y_label='Operating Margin (%)',
        trend_line=True
    )

    profit_chart_file = chart_path / "profitability_chart.png"
    chart_creator.save_chart(profit_chart, str(profit_chart_file))

    generator.create_two_column_slide(
        title="Profitability by Segment",
        left_content="""Segment Economics:

Enterprise:
• Revenue: $57.3M (45%)
• Gross Margin: 82%
• Operating Margin: 45%
• CAC: $18,400
• LTV: $420K

Mid-Market:
• Revenue: $44.6M (35%)
• Gross Margin: 76%
• Operating Margin: 32%
• CAC: $8,200
• LTV: $124K""",
        right_content="""SMB:
• Revenue: $25.4M (20%)
• Gross Margin: 68%
• Operating Margin: 18%
• CAC: $2,800
• LTV: $38K

Strategic Implications:
• Enterprise: High margin, scale
• Mid-Market: Balance growth
• SMB: Efficiency focus

Investment Allocation:
• Enterprise: 50% of sales
• Mid-Market: 35%
• SMB: 15% (automation)"""
    )

    generator.create_content_slide(
        title="Cash Flow & Runway",
        bullets=[
            "Operating cash flow: $38.2M (30% of revenue), strong collection",
            "Free cash flow: $29.7M after $8.5M capex investment",
            "Cash balance: $42.8M, 18-month runway at current burn rate"
        ],
        accent_bar=True
    )

    # ========================================================================
    # INSIGHTS & RECOMMENDATIONS
    # ========================================================================
    print("Creating section: Insights")
    generator.create_section_break("Key Insights & Recommendations")

    generator.create_content_slide(
        title="Data-Driven Insights",
        bullets=[
            "Enterprise segment drives 65% of profit on 45% of revenue: double down",
            "SMB churn 4.7% vs 1.1% enterprise: automation + self-service needed",
            "AI feature 52% adoption, 28% engagement lift: accelerate development"
        ],
        accent_bar=True
    )

    generator.create_two_column_slide(
        title="Strategic Recommendations",
        left_content="""Growth Initiatives:

1. Enterprise Expansion
   • Scale sales team by 40%
   • Invest in ABM campaigns
   • Target Fortune 1000

2. Product Development
   • Accelerate AI features
   • Mobile app enhancement
   • API marketplace launch

3. International Growth
   • EMEA expansion ($15M)
   • APAC pilot scaling
   • Localization investment""",
        right_content="""Optimization Focus:

1. SMB Segment
   • Self-service onboarding
   • Automated success programs
   • Pricing tier adjustment

2. Sales Efficiency
   • Partner channel growth
   • Sales automation tools
   • Predictive lead scoring

3. Cost Management
   • Infrastructure optimization
   • Support automation
   • Vendor consolidation"""
    )

    generator.create_content_slide(
        title="Next Quarter Focus",
        bullets=[
            "Revenue target: $145M (+14% QoQ) with enterprise focus",
            "Reduce SMB CAC by 25% through automation and self-service",
            "Launch AI product tier, target 75% adoption by existing customers"
        ],
        accent_bar=True
    )

    # ========================================================================
    # EXPORT
    # ========================================================================
    print("\nExporting presentation...")
    result = generator.export_for_html2pptx(output_path)

    print("=" * 70)
    print("Data Visualization Presentation Complete!")
    print("=" * 70)
    print(f"Output file: {result['output_path']}")
    print(f"Total slides: {result['slide_count']}")
    print(f"Charts generated: 5 (saved to {chart_path})")
    print(f"\nNext step: python -m html2pptx {output_path} data-visualization.pptx")

    return result


if __name__ == "__main__":
    create_data_visualization_presentation()
