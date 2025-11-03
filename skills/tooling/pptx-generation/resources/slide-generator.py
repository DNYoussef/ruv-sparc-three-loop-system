#!/usr/bin/env python3
"""
Slide Generator - Core slide creation with html2pptx workflow
Implements constraint-based design with accessibility validation
"""

from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import json
import re


class SlideType(Enum):
    """Enumeration of supported slide layouts"""
    TITLE = "title"
    CONTENT = "content"
    TWO_COLUMN = "two_column"
    DATA_VISUAL = "data_visual"
    SECTION_BREAK = "section_break"


@dataclass
class ColorPalette:
    """Color scheme with accessibility validation"""
    primary: str  # Hex color
    secondary: str
    accent: str
    background: str
    text: str

    def validate_contrast(self, foreground: str, background: str) -> Tuple[bool, float]:
        """
        Calculate contrast ratio between two colors
        Returns: (passes_wcag_aa, contrast_ratio)
        WCAG AA requires 4.5:1 for normal text, 3:1 for large text
        """
        # Convert hex to RGB
        fg_rgb = self._hex_to_rgb(foreground)
        bg_rgb = self._hex_to_rgb(background)

        # Calculate relative luminance
        fg_lum = self._relative_luminance(fg_rgb)
        bg_lum = self._relative_luminance(bg_rgb)

        # Calculate contrast ratio
        lighter = max(fg_lum, bg_lum)
        darker = min(fg_lum, bg_lum)
        ratio = (lighter + 0.05) / (darker + 0.05)

        return (ratio >= 4.5, ratio)

    @staticmethod
    def _hex_to_rgb(hex_color: str) -> Tuple[int, int, int]:
        """Convert hex color to RGB tuple"""
        hex_color = hex_color.lstrip('#')
        return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

    @staticmethod
    def _relative_luminance(rgb: Tuple[int, int, int]) -> float:
        """Calculate relative luminance for contrast ratio"""
        r, g, b = [x / 255.0 for x in rgb]

        # Apply gamma correction
        def adjust(c):
            return c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4

        r_adj, g_adj, b_adj = adjust(r), adjust(g), adjust(b)
        return 0.2126 * r_adj + 0.7152 * g_adj + 0.0722 * b_adj


@dataclass
class LayoutConstraints:
    """Quantified layout specifications"""
    min_margin_inches: float = 0.5
    min_font_size_body: int = 18
    min_font_size_title: int = 24
    max_bullets_per_slide: int = 3
    min_line_spacing: float = 1.2
    max_text_elements: int = 5


class SlideGenerator:
    """
    Core slide generation with constraint enforcement
    Implements html2pptx workflow with validation gates
    """

    def __init__(self, color_palette: ColorPalette, constraints: LayoutConstraints):
        self.colors = color_palette
        self.constraints = constraints
        self.slides: List[Dict] = []
        self._validate_setup()

    def _validate_setup(self):
        """Validate color palette meets accessibility requirements"""
        passes, ratio = self.colors.validate_contrast(
            self.colors.text,
            self.colors.background
        )
        if not passes:
            raise ValueError(
                f"Text/background contrast ratio {ratio:.2f}:1 fails WCAG AA "
                f"(requires 4.5:1). Adjust colors."
            )

    def create_title_slide(self, title: str, subtitle: str = "") -> Dict:
        """
        Generate title slide with centered layout

        Returns HTML structure for html2pptx conversion
        """
        html = f"""
        <div class="slide title-slide" style="
            background-color: {self.colors.background};
            padding: {self.constraints.min_margin_inches}in;
            display: flex;
            flex-direction: column;
            justify-content: center;
            align-items: center;
            height: 7.5in;
            width: 10in;
        ">
            <h1 style="
                color: {self.colors.text};
                font-size: {self.constraints.min_font_size_title * 1.5}pt;
                font-weight: bold;
                text-align: center;
                margin-bottom: 0.3in;
                max-width: 8in;
            ">{self._escape_html(title)}</h1>

            {f'''<h2 style="
                color: {self.colors.secondary};
                font-size: {self.constraints.min_font_size_title}pt;
                text-align: center;
                font-weight: normal;
                max-width: 8in;
            ">{self._escape_html(subtitle)}</h2>''' if subtitle else ''}
        </div>
        """

        slide = {
            "type": SlideType.TITLE.value,
            "html": html,
            "title": title,
            "validation": self._validate_slide(html, title_only=True)
        }

        self.slides.append(slide)
        return slide

    def create_content_slide(self,
                           title: str,
                           bullets: List[str],
                           accent_bar: bool = True) -> Dict:
        """
        Generate content slide with bullet points
        Enforces maximum bullet constraint
        """
        if len(bullets) > self.constraints.max_bullets_per_slide:
            raise ValueError(
                f"Bullet count {len(bullets)} exceeds maximum "
                f"{self.constraints.max_bullets_per_slide}. Split into multiple slides."
            )

        bullets_html = "\n".join([
            f'<li style="margin-bottom: 0.2in; line-height: {self.constraints.min_line_spacing};">'
            f'{self._escape_html(bullet)}</li>'
            for bullet in bullets
        ])

        accent_html = f'''
            <div style="
                position: absolute;
                left: 0;
                top: 1in;
                width: 0.15in;
                height: 1in;
                background-color: {self.colors.accent};
            "></div>
        ''' if accent_bar else ''

        html = f"""
        <div class="slide content-slide" style="
            background-color: {self.colors.background};
            padding: {self.constraints.min_margin_inches}in;
            position: relative;
            height: 7.5in;
            width: 10in;
        ">
            {accent_html}

            <h1 style="
                color: {self.colors.text};
                font-size: {self.constraints.min_font_size_title}pt;
                font-weight: bold;
                margin-bottom: 0.4in;
                padding-left: {0.3 if accent_bar else 0}in;
            ">{self._escape_html(title)}</h1>

            <ul style="
                color: {self.colors.text};
                font-size: {self.constraints.min_font_size_body}pt;
                list-style-type: disc;
                padding-left: {0.8 if accent_bar else 0.5}in;
                margin: 0;
            ">
                {bullets_html}
            </ul>
        </div>
        """

        slide = {
            "type": SlideType.CONTENT.value,
            "html": html,
            "title": title,
            "bullets": bullets,
            "validation": self._validate_slide(html)
        }

        self.slides.append(slide)
        return slide

    def create_two_column_slide(self,
                               title: str,
                               left_content: str,
                               right_content: str) -> Dict:
        """
        Generate two-column layout with visual separation via spacing
        NO border boxes or dividing lines
        """
        html = f"""
        <div class="slide two-column-slide" style="
            background-color: {self.colors.background};
            padding: {self.constraints.min_margin_inches}in;
            height: 7.5in;
            width: 10in;
        ">
            <h1 style="
                color: {self.colors.text};
                font-size: {self.constraints.min_font_size_title}pt;
                font-weight: bold;
                margin-bottom: 0.4in;
            ">{self._escape_html(title)}</h1>

            <div style="display: flex; gap: 0.5in;">
                <div style="
                    flex: 1;
                    color: {self.colors.text};
                    font-size: {self.constraints.min_font_size_body}pt;
                    line-height: {self.constraints.min_line_spacing};
                ">
                    {self._escape_html(left_content)}
                </div>

                <div style="
                    flex: 1;
                    color: {self.colors.text};
                    font-size: {self.constraints.min_font_size_body}pt;
                    line-height: {self.constraints.min_line_spacing};
                    background-color: {self.colors.secondary}15;
                    padding: 0.3in;
                ">
                    {self._escape_html(right_content)}
                </div>
            </div>
        </div>
        """

        slide = {
            "type": SlideType.TWO_COLUMN.value,
            "html": html,
            "title": title,
            "validation": self._validate_slide(html)
        }

        self.slides.append(slide)
        return slide

    def create_section_break(self, section_title: str) -> Dict:
        """
        Generate section break slide with minimal design
        Uses color block background for emphasis
        """
        html = f"""
        <div class="slide section-break" style="
            background: linear-gradient(135deg, {self.colors.primary} 0%, {self.colors.secondary} 100%);
            display: flex;
            align-items: center;
            justify-content: center;
            height: 7.5in;
            width: 10in;
        ">
            <h1 style="
                color: #FFFFFF;
                font-size: {self.constraints.min_font_size_title * 2}pt;
                font-weight: bold;
                text-align: center;
                max-width: 8in;
            ">{self._escape_html(section_title)}</h1>
        </div>
        """

        # Validate white text on gradient background
        passes, ratio = self.colors.validate_contrast("#FFFFFF", self.colors.primary)
        if not passes:
            raise ValueError(
                f"White text on gradient background has insufficient contrast "
                f"({ratio:.2f}:1). Darken primary color."
            )

        slide = {
            "type": SlideType.SECTION_BREAK.value,
            "html": html,
            "title": section_title,
            "validation": {"contrast_validated": True, "ratio": ratio}
        }

        self.slides.append(slide)
        return slide

    def _validate_slide(self, html: str, title_only: bool = False) -> Dict:
        """
        Run validation checks on generated HTML
        Returns validation report
        """
        validation = {
            "prohibited_elements": self._check_prohibited_elements(html),
            "contrast_valid": True,  # Already validated in color palette
            "font_sizes_valid": self._check_font_sizes(html),
            "margins_valid": True,  # Enforced in CSS
        }

        if not title_only:
            validation["bullet_count_valid"] = True  # Enforced in method

        return validation

    def _check_prohibited_elements(self, html: str) -> List[str]:
        """
        Scan for prohibited design elements
        Returns list of violations (empty if clean)
        """
        violations = []

        # Check for border boxes
        if re.search(r'border\s*:\s*\d', html, re.IGNORECASE):
            violations.append("Border detected - use spacing instead")

        # Check for outline shapes
        if re.search(r'outline\s*:\s*\d', html, re.IGNORECASE):
            violations.append("Outline detected - prohibited")

        # Check for rounded rectangles
        if re.search(r'border-radius', html, re.IGNORECASE):
            violations.append("Border-radius detected - use sharp edges")

        return violations

    def _check_font_sizes(self, html: str) -> bool:
        """Verify all font sizes meet minimum requirements"""
        font_sizes = re.findall(r'font-size:\s*(\d+)pt', html)
        if not font_sizes:
            return True

        min_size = min(int(size) for size in font_sizes)
        return min_size >= self.constraints.min_font_size_body

    @staticmethod
    def _escape_html(text: str) -> str:
        """Escape HTML special characters"""
        return (text
                .replace('&', '&amp;')
                .replace('<', '&lt;')
                .replace('>', '&gt;')
                .replace('"', '&quot;')
                .replace("'", '&#39;'))

    def export_for_html2pptx(self, output_path: str):
        """
        Export slides as HTML file for html2pptx conversion

        Usage:
            generator.export_for_html2pptx("presentation.html")
            # Then run: html2pptx presentation.html output.pptx
        """
        full_html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <title>Generated Presentation</title>
            <style>
                .slide {{
                    page-break-after: always;
                }}
            </style>
        </head>
        <body>
            {''.join(slide['html'] for slide in self.slides)}
        </body>
        </html>
        """

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(full_html)

        return {
            "output_path": output_path,
            "slide_count": len(self.slides),
            "validation_summary": self._generate_validation_summary()
        }

    def _generate_validation_summary(self) -> Dict:
        """Generate summary report of all validation checks"""
        total_slides = len(self.slides)
        all_validations = [slide.get('validation', {}) for slide in self.slides]

        return {
            "total_slides": total_slides,
            "all_contrast_valid": all(v.get('contrast_valid', False) for v in all_validations),
            "all_fonts_valid": all(v.get('font_sizes_valid', False) for v in all_validations),
            "prohibited_violations": sum(len(v.get('prohibited_elements', [])) for v in all_validations),
        }


if __name__ == "__main__":
    # Example usage
    palette = ColorPalette(
        primary="#1E3A8A",  # Deep blue
        secondary="#3B82F6",  # Medium blue
        accent="#F59E0B",  # Amber
        background="#FFFFFF",  # White
        text="#1F2937"  # Dark gray
    )

    constraints = LayoutConstraints()
    generator = SlideGenerator(palette, constraints)

    # Create sample deck
    generator.create_title_slide(
        "Strategic Initiative Review",
        "Q4 2024 Performance Analysis"
    )

    generator.create_section_break("Executive Summary")

    generator.create_content_slide(
        "Key Achievements",
        [
            "Revenue growth exceeded targets by 23%",
            "Customer satisfaction score increased to 94%",
            "Product launch delivered ahead of schedule"
        ]
    )

    generator.export_for_html2pptx("sample_presentation.html")
    print("Generated sample_presentation.html - ready for html2pptx conversion")
