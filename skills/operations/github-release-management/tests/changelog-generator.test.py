#!/usr/bin/env python3
"""
Tests for Changelog Generator
Validates changelog parsing, categorization, and formatting
"""

import unittest
import tempfile
import shutil
import subprocess
import json
from pathlib import Path
import sys
import os

# Add parent directory to path to import module
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'resources'))

try:
    from changelog_generator import Commit, ChangelogGenerator, CATEGORIES
except ImportError:
    # If module import fails, skip tests
    print("Warning: Could not import changelog_generator module")
    Commit = None
    ChangelogGenerator = None
    CATEGORIES = {}


class TestCommitParsing(unittest.TestCase):
    """Test conventional commit parsing"""

    def setUp(self):
        if Commit is None:
            self.skipTest("changelog_generator module not available")

    def test_simple_feature_commit(self):
        """Test parsing simple feature commit"""
        commit = Commit(
            sha='abc123',
            message='feat: add new feature',
            author='Alice',
            date='2024-01-15'
        )

        self.assertEqual(commit.category, 'feature')
        self.assertEqual(commit.description, 'add new feature')
        self.assertIsNone(commit.scope)
        self.assertFalse(commit.breaking)

    def test_scoped_commit(self):
        """Test parsing scoped commit"""
        commit = Commit(
            sha='def456',
            message='fix(api): correct validation error',
            author='Bob',
            date='2024-01-15'
        )

        self.assertEqual(commit.category, 'fix')
        self.assertEqual(commit.scope, 'api')
        self.assertEqual(commit.description, 'correct validation error')

    def test_breaking_change_exclamation(self):
        """Test breaking change with exclamation mark"""
        commit = Commit(
            sha='ghi789',
            message='feat!: remove deprecated API',
            author='Charlie',
            date='2024-01-15'
        )

        self.assertEqual(commit.category, 'breaking')
        self.assertTrue(commit.breaking)

    def test_breaking_change_footer(self):
        """Test breaking change in footer"""
        commit = Commit(
            sha='jkl012',
            message='feat: new API\n\nBREAKING CHANGE: removed old endpoints',
            author='Diana',
            date='2024-01-15'
        )

        self.assertEqual(commit.category, 'breaking')
        self.assertTrue(commit.breaking)

    def test_pr_number_extraction(self):
        """Test PR number extraction from commit message"""
        commit = Commit(
            sha='mno345',
            message='fix: bug fix (#123)',
            author='Eve',
            date='2024-01-15'
        )

        self.assertEqual(commit.pr_number, '123')

    def test_commit_type_mapping(self):
        """Test commit type to category mapping"""
        test_cases = [
            ('feat: feature', 'feature'),
            ('fix: bug fix', 'fix'),
            ('docs: documentation', 'docs'),
            ('refactor: code refactor', 'refactor'),
            ('perf: optimization', 'performance'),
            ('test: add tests', 'test'),
            ('build: build change', 'build'),
            ('ci: ci change', 'ci'),
            ('chore: maintenance', 'chore'),
            ('security: security fix', 'security'),
        ]

        for message, expected_category in test_cases:
            with self.subTest(message=message):
                commit = Commit('sha', message, 'author', 'date')
                self.assertEqual(commit.category, expected_category)

    def test_non_conventional_commit(self):
        """Test non-conventional commit defaults to 'other'"""
        commit = Commit(
            sha='pqr678',
            message='Update README',
            author='Frank',
            date='2024-01-15'
        )

        self.assertEqual(commit.category, 'other')
        self.assertEqual(commit.description, 'Update README')


class TestChangelogGeneration(unittest.TestCase):
    """Test changelog generation with real git repository"""

    def setUp(self):
        if ChangelogGenerator is None:
            self.skipTest("changelog_generator module not available")

        # Create temporary git repository
        self.test_dir = tempfile.mkdtemp()
        self.repo_path = Path(self.test_dir)

        # Initialize git repo
        subprocess.run(['git', 'init'], cwd=self.repo_path, check=True, capture_output=True)
        subprocess.run(
            ['git', 'config', 'user.email', 'test@example.com'],
            cwd=self.repo_path,
            check=True,
            capture_output=True
        )
        subprocess.run(
            ['git', 'config', 'user.name', 'Test User'],
            cwd=self.repo_path,
            check=True,
            capture_output=True
        )

        # Create initial commit
        (self.repo_path / 'README.md').write_text('# Test')
        subprocess.run(['git', 'add', '.'], cwd=self.repo_path, check=True, capture_output=True)
        subprocess.run(
            ['git', 'commit', '-m', 'Initial commit'],
            cwd=self.repo_path,
            check=True,
            capture_output=True
        )
        subprocess.run(
            ['git', 'tag', 'v1.0.0'],
            cwd=self.repo_path,
            check=True,
            capture_output=True
        )

    def tearDown(self):
        # Clean up temporary directory
        if hasattr(self, 'test_dir') and os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def create_commit(self, message: str, filename: str = None):
        """Helper to create a commit"""
        if filename is None:
            filename = f'file_{len(os.listdir(self.repo_path))}.txt'

        filepath = self.repo_path / filename
        filepath.write_text('content')

        subprocess.run(['git', 'add', '.'], cwd=self.repo_path, check=True, capture_output=True)
        subprocess.run(
            ['git', 'commit', '-m', message],
            cwd=self.repo_path,
            check=True,
            capture_output=True
        )

    def test_get_commits(self):
        """Test getting commits between refs"""
        self.create_commit('feat: add feature 1')
        self.create_commit('fix: bug fix 1')
        self.create_commit('docs: update docs')

        generator = ChangelogGenerator('v1.0.0', 'HEAD', str(self.repo_path))
        commits = generator.get_commits()

        self.assertEqual(len(commits), 3)

    def test_categorization(self):
        """Test commit categorization"""
        self.create_commit('feat: new feature')
        self.create_commit('fix: bug fix')
        self.create_commit('docs: documentation')

        generator = ChangelogGenerator('v1.0.0', 'HEAD', str(self.repo_path))
        generator.get_commits()

        self.assertEqual(len(generator.categories['feature']), 1)
        self.assertEqual(len(generator.categories['fix']), 1)
        self.assertEqual(len(generator.categories['docs']), 1)

    def test_markdown_generation(self):
        """Test markdown changelog generation"""
        self.create_commit('feat: add authentication')
        self.create_commit('fix: correct validation')
        self.create_commit('docs: update API docs')

        generator = ChangelogGenerator('v1.0.0', 'HEAD', str(self.repo_path))
        generator.get_commits()

        markdown = generator.generate_markdown(version='2.0.0', use_emoji=True)

        self.assertIn('## 2.0.0', markdown)
        self.assertIn('🚀 Features', markdown)
        self.assertIn('add authentication', markdown)
        self.assertIn('🐛 Bug Fixes', markdown)
        self.assertIn('correct validation', markdown)
        self.assertIn('📚 Documentation', markdown)
        self.assertIn('update API docs', markdown)

    def test_json_generation(self):
        """Test JSON changelog generation"""
        self.create_commit('feat: new feature')

        generator = ChangelogGenerator('v1.0.0', 'HEAD', str(self.repo_path))
        generator.get_commits()

        json_output = generator.generate_json()
        data = json.loads(json_output)

        self.assertIn('categories', data)
        self.assertIn('feature', data['categories'])
        self.assertEqual(len(data['categories']['feature']['commits']), 1)

    def test_breaking_change_detection(self):
        """Test breaking change detection"""
        self.create_commit('feat!: breaking API change')

        generator = ChangelogGenerator('v1.0.0', 'HEAD', str(self.repo_path))
        generator.get_commits()

        self.assertEqual(len(generator.categories['breaking']), 1)

    def test_migration_detection(self):
        """Test migration guide detection"""
        self.create_commit('feat!: remove old API')
        self.create_commit('feat: add database migration')

        generator = ChangelogGenerator('v1.0.0', 'HEAD', str(self.repo_path))
        generator.get_commits()

        migrations = generator.detect_migration_needed()

        self.assertGreater(len(migrations), 0)
        self.assertTrue(any('Breaking changes' in m for m in migrations))
        self.assertTrue(any('Database' in m for m in migrations))

    def test_category_priority_ordering(self):
        """Test categories are ordered by priority"""
        self.create_commit('docs: update docs')
        self.create_commit('feat: new feature')
        self.create_commit('feat!: breaking change')
        self.create_commit('fix: bug fix')

        generator = ChangelogGenerator('v1.0.0', 'HEAD', str(self.repo_path))
        generator.get_commits()

        markdown = generator.generate_markdown()

        # Breaking should come first
        breaking_pos = markdown.find('💥 Breaking Changes')
        feature_pos = markdown.find('🚀 Features')
        fix_pos = markdown.find('🐛 Bug Fixes')
        docs_pos = markdown.find('📚 Documentation')

        self.assertLess(breaking_pos, feature_pos)
        self.assertLess(feature_pos, fix_pos)
        self.assertLess(fix_pos, docs_pos)

    def test_contributor_list(self):
        """Test contributor list generation"""
        self.create_commit('feat: feature by Alice')

        generator = ChangelogGenerator('v1.0.0', 'HEAD', str(self.repo_path))
        generator.get_commits()

        markdown = generator.generate_markdown(include_contributors=True)

        self.assertIn('Contributors', markdown)
        self.assertIn('Test User', markdown)

    def test_no_emoji_mode(self):
        """Test changelog without emojis"""
        self.create_commit('feat: new feature')

        generator = ChangelogGenerator('v1.0.0', 'HEAD', str(self.repo_path))
        generator.get_commits()

        markdown = generator.generate_markdown(use_emoji=False)

        self.assertNotIn('🚀', markdown)
        self.assertIn('Features', markdown)


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and error handling"""

    def setUp(self):
        if ChangelogGenerator is None:
            self.skipTest("changelog_generator module not available")

    def test_empty_commit_range(self):
        """Test handling empty commit range"""
        test_dir = tempfile.mkdtemp()

        try:
            subprocess.run(['git', 'init'], cwd=test_dir, check=True, capture_output=True)
            subprocess.run(
                ['git', 'config', 'user.email', 'test@example.com'],
                cwd=test_dir,
                check=True,
                capture_output=True
            )
            subprocess.run(
                ['git', 'config', 'user.name', 'Test'],
                cwd=test_dir,
                check=True,
                capture_output=True
            )

            (Path(test_dir) / 'README.md').write_text('Test')
            subprocess.run(['git', 'add', '.'], cwd=test_dir, check=True, capture_output=True)
            subprocess.run(
                ['git', 'commit', '-m', 'Initial'],
                cwd=test_dir,
                check=True,
                capture_output=True
            )
            subprocess.run(['git', 'tag', 'v1.0.0'], cwd=test_dir, check=True, capture_output=True)

            # No commits after tag
            generator = ChangelogGenerator('v1.0.0', 'HEAD', test_dir)
            commits = generator.get_commits()

            self.assertEqual(len(commits), 0)

        finally:
            shutil.rmtree(test_dir)


if __name__ == '__main__':
    unittest.main()
