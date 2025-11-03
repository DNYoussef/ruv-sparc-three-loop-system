#!/usr/bin/env python3
"""
Test Suite for Slide Generator
Validates constraint enforcement, accessibility, and design principles
"""

import unittest
import sys
import os
from io import StringIO

# Add resources directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'resources'))

from slide_generator import (
    SlideGenerator,
    ColorPalette,
    LayoutConstraints,
    SlideType
)


class TestColorPalette(unittest.TestCase):
    """Test color palette and contrast validation"""

    def test_valid_contrast_passes(self):
        """High contrast text/background should pass WCAG AA"""
        palette = ColorPalette(
            primary="#1E3A8A",
            secondary="#3B82F6",
            accent="#F59E0B",
            background="#FFFFFF",
            text="#1F2937"
        )

        passes, ratio = palette.validate_contrast("#1F2937", "#FFFFFF")
        self.assertTrue(passes)
        self.assertGreaterEqual(ratio, 4.5)

    def test_low_contrast_fails(self):
        """Low contrast should fail WCAG AA"""
        palette = ColorPalette(
            primary="#1E3A8A",
            secondary="#3B82F6",
            accent="#F59E0B",
            background="#FFFFFF",
            text="#CCCCCC"  # Light gray on white
        )

        passes, ratio = palette.validate_contrast("#CCCCCC", "#FFFFFF")
        self.assertFalse(passes)
        self.assertLess(ratio, 4.5)

    def test_contrast_ratio_calculation(self):
        """Verify contrast ratio calculations"""
        palette = ColorPalette(
            primary="#000000",
            secondary="#FFFFFF",
            accent="#FF0000",
            background="#FFFFFF",
            text="#000000"
        )

        # Black on white should be maximum contrast (21:1)
        passes, ratio = palette.validate_contrast("#000000", "#FFFFFF")
        self.assertTrue(passes)
        self.assertGreater(ratio, 20.0)

    def test_hex_to_rgb_conversion(self):
        """Test hex color conversion"""
        palette = ColorPalette(
            primary="#FF0000",
            secondary="#00FF00",
            accent="#0000FF",
            background="#FFFFFF",
            text="#000000"
        )

        # Test red
        self.assertEqual(palette._hex_to_rgb("#FF0000"), (255, 0, 0))
        # Test green
        self.assertEqual(palette._hex_to_rgb("#00FF00"), (0, 255, 0))
        # Test blue
        self.assertEqual(palette._hex_to_rgb("#0000FF"), (0, 0, 255))


class TestLayoutConstraints(unittest.TestCase):
    """Test layout constraint validation"""

    def test_default_constraints(self):
        """Verify default constraint values"""
        constraints = LayoutConstraints()

        self.assertEqual(constraints.min_margin_inches, 0.5)
        self.assertEqual(constraints.min_font_size_body, 18)
        self.assertEqual(constraints.min_font_size_title, 24)
        self.assertEqual(constraints.max_bullets_per_slide, 3)
        self.assertEqual(constraints.min_line_spacing, 1.2)

    def test_custom_constraints(self):
        """Custom constraints should override defaults"""
        constraints = LayoutConstraints(
            min_margin_inches=0.75,
            min_font_size_body=20,
            max_bullets_per_slide=5
        )

        self.assertEqual(constraints.min_margin_inches, 0.75)
        self.assertEqual(constraints.min_font_size_body, 20)
        self.assertEqual(constraints.max_bullets_per_slide, 5)


class TestSlideGenerator(unittest.TestCase):
    """Test slide generation and validation"""

    def setUp(self):
        """Create test palette and constraints"""
        self.palette = ColorPalette(
            primary="#1E3A8A",
            secondary="#3B82F6",
            accent="#F59E0B",
            background="#FFFFFF",
            text="#1F2937"
        )
        self.constraints = LayoutConstraints()
        self.generator = SlideGenerator(self.palette, self.constraints)

    def test_generator_initialization(self):
        """Generator should initialize with valid palette"""
        self.assertEqual(len(self.generator.slides), 0)
        self.assertEqual(self.generator.colors.primary, "#1E3A8A")

    def test_invalid_palette_rejected(self):
        """Generator should reject low-contrast palettes"""
        bad_palette = ColorPalette(
            primary="#CCCCCC",
            secondary="#DDDDDD",
            accent="#EEEEEE",
            background="#FFFFFF",
            text="#CCCCCC"  # Fails contrast check
        )

        with self.assertRaises(ValueError) as context:
            SlideGenerator(bad_palette, self.constraints)

        self.assertIn("contrast ratio", str(context.exception).lower())

    def test_create_title_slide(self):
        """Title slide should generate valid HTML"""
        slide = self.generator.create_title_slide(
            "Test Presentation",
            "Subtitle Text"
        )

        self.assertEqual(slide['type'], SlideType.TITLE.value)
        self.assertEqual(slide['title'], "Test Presentation")
        self.assertIn("Test Presentation", slide['html'])
        self.assertIn("Subtitle Text", slide['html'])

        # Verify validation passed
        self.assertTrue(slide['validation']['contrast_valid'])

    def test_create_content_slide(self):
        """Content slide should enforce bullet limit"""
        bullets = ["Point 1", "Point 2", "Point 3"]
        slide = self.generator.create_content_slide(
            "Content Slide",
            bullets
        )

        self.assertEqual(slide['type'], SlideType.CONTENT.value)
        self.assertEqual(slide['bullets'], bullets)
        self.assertIn("Point 1", slide['html'])
        self.assertIn("Point 3", slide['html'])

    def test_content_slide_rejects_too_many_bullets(self):
        """Should reject slides with >3 bullets"""
        too_many_bullets = ["Point 1", "Point 2", "Point 3", "Point 4"]

        with self.assertRaises(ValueError) as context:
            self.generator.create_content_slide(
                "Invalid Slide",
                too_many_bullets
            )

        self.assertIn("exceeds maximum", str(context.exception).lower())

    def test_two_column_slide(self):
        """Two-column layout should create proper structure"""
        slide = self.generator.create_two_column_slide(
            "Comparison",
            "Left content here",
            "Right content here"
        )

        self.assertEqual(slide['type'], SlideType.TWO_COLUMN.value)
        self.assertIn("Left content here", slide['html'])
        self.assertIn("Right content here", slide['html'])
        self.assertIn("display: flex", slide['html'])

    def test_section_break_slide(self):
        """Section break should use gradient background"""
        slide = self.generator.create_section_break("New Section")

        self.assertEqual(slide['type'], SlideType.SECTION_BREAK.value)
        self.assertIn("New Section", slide['html'])
        self.assertIn("gradient", slide['html'].lower())

    def test_prohibited_elements_detection(self):
        """Should detect prohibited design elements"""
        # HTML with border
        html_with_border = '<div style="border: 1px solid black;">Content</div>'
        violations = self.generator._check_prohibited_elements(html_with_border)
        self.assertGreater(len(violations), 0)
        self.assertTrue(any('border' in v.lower() for v in violations))

        # Clean HTML
        clean_html = '<div style="padding: 10px;">Content</div>'
        violations = self.generator._check_prohibited_elements(clean_html)
        self.assertEqual(len(violations), 0)

    def test_font_size_validation(self):
        """Should validate minimum font sizes"""
        # Valid HTML
        valid_html = '<p style="font-size: 18pt;">Text</p>'
        self.assertTrue(self.generator._check_font_sizes(valid_html))

        # Invalid HTML
        invalid_html = '<p style="font-size: 12pt;">Text</p>'
        self.assertFalse(self.generator._check_font_sizes(invalid_html))

    def test_html_escaping(self):
        """Should properly escape HTML special characters"""
        text = '<script>alert("xss")</script>'
        escaped = self.generator._escape_html(text)

        self.assertNotIn('<script>', escaped)
        self.assertIn('&lt;script&gt;', escaped)

    def test_export_html(self):
        """Should export complete HTML document"""
        self.generator.create_title_slide("Test", "Subtitle")
        self.generator.create_content_slide(
            "Content",
            ["Bullet 1", "Bullet 2"]
        )

        # Export to temporary file
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.html', delete=False) as f:
            temp_path = f.name

        try:
            result = self.generator.export_for_html2pptx(temp_path)

            self.assertEqual(result['slide_count'], 2)
            self.assertTrue(os.path.exists(temp_path))

            # Verify HTML structure
            with open(temp_path, 'r', encoding='utf-8') as f:
                html_content = f.read()

            self.assertIn('<!DOCTYPE html>', html_content)
            self.assertIn('Test', html_content)
            self.assertIn('Bullet 1', html_content)

        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    def test_validation_summary(self):
        """Validation summary should aggregate results"""
        self.generator.create_title_slide("Title", "Subtitle")
        self.generator.create_content_slide("Content", ["A", "B", "C"])

        summary = self.generator._generate_validation_summary()

        self.assertEqual(summary['total_slides'], 2)
        self.assertTrue(summary['all_contrast_valid'])
        self.assertTrue(summary['all_fonts_valid'])
        self.assertEqual(summary['prohibited_violations'], 0)


class TestIntegration(unittest.TestCase):
    """Integration tests for complete workflows"""

    def test_full_deck_generation(self):
        """Generate complete multi-slide deck"""
        palette = ColorPalette(
            primary="#1E3A8A",
            secondary="#3B82F6",
            accent="#F59E0B",
            background="#FFFFFF",
            text="#1F2937"
        )
        constraints = LayoutConstraints()
        generator = SlideGenerator(palette, constraints)

        # Create full deck
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

        generator.create_two_column_slide(
            "Financial Overview",
            "Q4 revenue reached $12.5M, representing 23% YoY growth.",
            "Operating margin improved to 32%, driven by efficiency gains."
        )

        # Verify deck integrity
        self.assertEqual(len(generator.slides), 4)

        # Export and validate
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.html', delete=False) as f:
            temp_path = f.name

        try:
            result = generator.export_for_html2pptx(temp_path)
            self.assertTrue(result['validation_summary']['all_contrast_valid'])
            self.assertEqual(result['validation_summary']['prohibited_violations'], 0)

        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)


def run_tests():
    """Run all tests and report results"""
    # Suppress print output during tests
    suite = unittest.TestLoader().loadTestsFromModule(sys.modules[__name__])
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    print("\n" + "="*70)
    print("Test Summary:")
    print("="*70)
    print(f"Tests run: {result.testsRun}")
    print(f"Successes: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")

    if result.wasSuccessful():
        print("\n✓ All tests passed!")
        return 0
    else:
        print("\n✗ Some tests failed")
        return 1


if __name__ == '__main__':
    exit_code = run_tests()
    sys.exit(exit_code)
