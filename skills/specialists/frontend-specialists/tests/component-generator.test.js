/**
 * Component Generator Tests
 *
 * Comprehensive test suite for component-generator.js
 *
 * @author Frontend Specialists Team
 * @version 1.0.0
 */

const fs = require('fs');
const path = require('path');
const { ComponentGenerator } = require('../resources/scripts/component-generator');

// Mock fs module
jest.mock('fs');

describe('ComponentGenerator', () => {
  let mockFs;

  beforeEach(() => {
    // Reset mocks
    jest.clearAllMocks();

    // Setup fs mocks
    mockFs = {
      existsSync: jest.fn().mockReturnValue(false),
      mkdirSync: jest.fn(),
      writeFileSync: jest.fn(),
    };

    fs.existsSync = mockFs.existsSync;
    fs.mkdirSync = mockFs.mkdirSync;
    fs.writeFileSync = mockFs.writeFileSync;
  });

  describe('Constructor', () => {
    it('should initialize with default options', () => {
      const generator = new ComponentGenerator({ name: 'Button' });

      expect(generator.framework).toBe('react');
      expect(generator.name).toBe('Button');
      expect(generator.type).toBe('functional');
      expect(generator.typescript).toBe(true);
      expect(generator.generateTests).toBe(true);
    });

    it('should accept custom options', () => {
      const generator = new ComponentGenerator({
        framework: 'vue',
        name: 'Card',
        type: 'composition',
        typescript: false,
        tests: false,
        storybook: true,
      });

      expect(generator.framework).toBe('vue');
      expect(generator.name).toBe('Card');
      expect(generator.type).toBe('composition');
      expect(generator.typescript).toBe(false);
      expect(generator.generateTests).toBe(false);
      expect(generator.generateStorybook).toBe(true);
    });
  });

  describe('generate()', () => {
    it('should create component directory', () => {
      const generator = new ComponentGenerator({
        name: 'Button',
        outputDir: './src/components',
      });

      // Suppress console.log
      jest.spyOn(console, 'log').mockImplementation(() => {});

      generator.generate();

      expect(mockFs.mkdirSync).toHaveBeenCalledWith(
        './src/components/Button',
        { recursive: true }
      );
    });

    it('should generate all files for React component', () => {
      const generator = new ComponentGenerator({
        name: 'Button',
        framework: 'react',
        tests: true,
        storybook: true,
      });

      jest.spyOn(console, 'log').mockImplementation(() => {});

      generator.generate();

      // Should create 5 files: component, styles, test, story, index
      expect(mockFs.writeFileSync).toHaveBeenCalledTimes(5);
    });

    it('should generate only component and styles when tests disabled', () => {
      const generator = new ComponentGenerator({
        name: 'Button',
        tests: false,
        storybook: false,
      });

      jest.spyOn(console, 'log').mockImplementation(() => {});

      generator.generate();

      // Should create 3 files: component, styles, index
      expect(mockFs.writeFileSync).toHaveBeenCalledTimes(3);
    });
  });

  describe('generateReactComponent()', () => {
    it('should generate TypeScript component by default', () => {
      const generator = new ComponentGenerator({ name: 'Button' });
      const componentDir = './src/components/Button';

      jest.spyOn(console, 'log').mockImplementation(() => {});

      generator.generateReactComponent(componentDir);

      const expectedPath = path.join(componentDir, 'Button.tsx');
      expect(mockFs.writeFileSync).toHaveBeenCalledWith(
        expectedPath,
        expect.stringContaining('export interface ButtonProps')
      );
    });

    it('should generate JavaScript component when typescript=false', () => {
      const generator = new ComponentGenerator({
        name: 'Button',
        typescript: false,
      });
      const componentDir = './src/components/Button';

      jest.spyOn(console, 'log').mockImplementation(() => {});

      generator.generateReactComponent(componentDir);

      const expectedPath = path.join(componentDir, 'Button.jsx');
      expect(mockFs.writeFileSync).toHaveBeenCalledWith(
        expectedPath,
        expect.stringContaining('export const Button')
      );
    });

    it('should include accessibility attributes', () => {
      const generator = new ComponentGenerator({ name: 'Button' });
      const componentDir = './src/components/Button';

      jest.spyOn(console, 'log').mockImplementation(() => {});

      generator.generateReactComponent(componentDir);

      const [, template] = mockFs.writeFileSync.mock.calls[0];
      expect(template).toContain('aria-label');
      expect(template).toContain('aria-disabled');
      expect(template).toContain('role="region"');
      expect(template).toContain('tabIndex');
    });
  });

  describe('generateVueComponent()', () => {
    it('should generate Vue component with Composition API', () => {
      const generator = new ComponentGenerator({
        name: 'Card',
        framework: 'vue',
      });
      const componentDir = './src/components/Card';

      jest.spyOn(console, 'log').mockImplementation(() => {});

      generator.generateVueComponent(componentDir);

      const [, template] = mockFs.writeFileSync.mock.calls[0];
      expect(template).toContain('defineComponent');
      expect(template).toContain('setup(props, { emit })');
      expect(template).toContain('<template>');
      expect(template).toContain('<script lang="ts">');
    });
  });

  describe('generateStyles()', () => {
    it('should generate CSS module', () => {
      const generator = new ComponentGenerator({ name: 'Button' });
      const componentDir = './src/components/Button';

      jest.spyOn(console, 'log').mockImplementation(() => {});

      generator.generateStyles(componentDir);

      const expectedPath = path.join(componentDir, 'Button.module.css');
      const [, template] = mockFs.writeFileSync.mock.calls[0];

      expect(mockFs.writeFileSync).toHaveBeenCalledWith(
        expectedPath,
        expect.any(String)
      );
      expect(template).toContain('.button {');
      expect(template).toContain('.button--primary');
      expect(template).toContain('.button--disabled');
      expect(template).toContain('@media (max-width: 768px)');
    });
  });

  describe('generateTest()', () => {
    it('should generate React test file', () => {
      const generator = new ComponentGenerator({
        name: 'Button',
        framework: 'react',
      });
      const componentDir = './src/components/Button';

      jest.spyOn(console, 'log').mockImplementation(() => {});

      generator.generateTest(componentDir);

      const expectedPath = path.join(componentDir, 'Button.test.tsx');
      const [, template] = mockFs.writeFileSync.mock.calls[0];

      expect(mockFs.writeFileSync).toHaveBeenCalledWith(
        expectedPath,
        expect.any(String)
      );
      expect(template).toContain('import { render, screen, fireEvent }');
      expect(template).toContain('describe(\'Button\'');
      expect(template).toContain('it(\'renders children correctly\'');
      expect(template).toContain('it(\'handles click events\'');
      expect(template).toContain('it(\'has correct accessibility attributes\'');
    });

    it('should generate Vue test file', () => {
      const generator = new ComponentGenerator({
        name: 'Card',
        framework: 'vue',
      });
      const componentDir = './src/components/Card';

      jest.spyOn(console, 'log').mockImplementation(() => {});

      generator.generateTest(componentDir);

      const [, template] = mockFs.writeFileSync.mock.calls[0];

      expect(template).toContain('import { mount }');
      expect(template).toContain('describe(\'Card\'');
      expect(template).toContain('it(\'renders slot content correctly\'');
    });
  });

  describe('generateStory()', () => {
    it('should generate Storybook story', () => {
      const generator = new ComponentGenerator({
        name: 'Button',
        storybook: true,
      });
      const componentDir = './src/components/Button';

      jest.spyOn(console, 'log').mockImplementation(() => {});

      generator.generateStory(componentDir);

      const expectedPath = path.join(componentDir, 'Button.stories.tsx');
      const [, template] = mockFs.writeFileSync.mock.calls[0];

      expect(mockFs.writeFileSync).toHaveBeenCalledWith(
        expectedPath,
        expect.any(String)
      );
      expect(template).toContain('import type { Meta, StoryObj }');
      expect(template).toContain('const meta: Meta<typeof Button>');
      expect(template).toContain('export const Primary: Story');
      expect(template).toContain('export const Disabled: Story');
    });
  });

  describe('generateIndex()', () => {
    it('should generate index file with exports', () => {
      const generator = new ComponentGenerator({ name: 'Button' });
      const componentDir = './src/components/Button';

      jest.spyOn(console, 'log').mockImplementation(() => {});

      generator.generateIndex(componentDir);

      const expectedPath = path.join(componentDir, 'index.ts');
      const [, template] = mockFs.writeFileSync.mock.calls[0];

      expect(mockFs.writeFileSync).toHaveBeenCalledWith(
        expectedPath,
        expect.any(String)
      );
      expect(template).toContain('export { Button }');
      expect(template).toContain('export type { ButtonProps }');
    });
  });

  describe('Error Handling', () => {
    it('should throw error for unsupported framework', () => {
      const generator = new ComponentGenerator({
        name: 'Button',
        framework: 'angular',
      });

      jest.spyOn(console, 'log').mockImplementation(() => {});

      expect(() => generator.generate()).toThrow(
        'Unsupported framework: angular'
      );
    });
  });

  describe('Templates', () => {
    it('should include proper TypeScript types in React template', () => {
      const generator = new ComponentGenerator({ name: 'Button' });
      const template = generator.reactTSTemplate();

      expect(template).toContain('export interface ButtonProps');
      expect(template).toContain('React.FC<ButtonProps>');
      expect(template).toContain('children?: React.ReactNode');
    });

    it('should include accessibility in all templates', () => {
      const generator = new ComponentGenerator({ name: 'Button' });

      const reactTemplate = generator.reactTSTemplate();
      expect(reactTemplate).toContain('aria-label');
      expect(reactTemplate).toContain('aria-disabled');
      expect(reactTemplate).toContain('tabIndex');

      const vueTemplate = generator.vueTemplate();
      expect(vueTemplate).toContain('aria-label');
      expect(vueTemplate).toContain('aria-disabled');
      expect(vueTemplate).toContain('tabindex');
    });

    it('should include keyboard navigation in templates', () => {
      const generator = new ComponentGenerator({ name: 'Button' });

      const reactTemplate = generator.reactTSTemplate();
      expect(reactTemplate).toContain('onKeyDown');
      expect(reactTemplate).toContain('key === \'Enter\'');

      const vueTemplate = generator.vueTemplate();
      expect(vueTemplate).toContain('@keydown.enter');
      expect(vueTemplate).toContain('@keydown.space');
    });
  });
});
