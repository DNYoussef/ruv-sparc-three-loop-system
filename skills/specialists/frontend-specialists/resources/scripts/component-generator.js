#!/usr/bin/env node

/**
 * Component Generator
 *
 * Scaffolds frontend components with best practices:
 * - TypeScript support
 * - Test file generation
 * - Storybook story creation
 * - Accessibility defaults (ARIA, semantic HTML)
 * - CSS module integration
 *
 * Usage:
 *   node component-generator.js --framework react --name Button --type functional
 *   node component-generator.js --framework vue --name Card --type composition
 *   node component-generator.js --framework react --name DataTable --tests --storybook
 *
 * @author Frontend Specialists Team
 * @version 1.0.0
 */

const fs = require('fs');
const path = require('path');

class ComponentGenerator {
  constructor(options) {
    this.framework = options.framework || 'react';
    this.name = options.name;
    this.type = options.type || 'functional';
    this.generateTests = options.tests !== false;
    this.generateStorybook = options.storybook || false;
    this.outputDir = options.outputDir || './src/components';
    this.typescript = options.typescript !== false;
  }

  /**
   * Generate component files
   */
  generate() {
    console.log(`\n🚀 Generating ${this.framework} component: ${this.name}\n`);

    const componentDir = path.join(this.outputDir, this.name);

    // Create component directory
    if (!fs.existsSync(componentDir)) {
      fs.mkdirSync(componentDir, { recursive: true });
    }

    // Generate main component file
    this.generateComponent(componentDir);

    // Generate styles
    this.generateStyles(componentDir);

    // Generate tests
    if (this.generateTests) {
      this.generateTest(componentDir);
    }

    // Generate Storybook story
    if (this.generateStorybook) {
      this.generateStory(componentDir);
    }

    // Generate index file
    this.generateIndex(componentDir);

    console.log(`\n✅ Component generated successfully at ${componentDir}\n`);
    this.printUsage();
  }

  /**
   * Generate React component
   */
  generateReactComponent(componentDir) {
    const ext = this.typescript ? 'tsx' : 'jsx';
    const filename = `${this.name}.${ext}`;
    const filepath = path.join(componentDir, filename);

    const template = this.typescript ? this.reactTSTemplate() : this.reactJSTemplate();
    fs.writeFileSync(filepath, template);
    console.log(`✓ Created ${filename}`);
  }

  /**
   * Generate Vue component
   */
  generateVueComponent(componentDir) {
    const ext = this.typescript ? 'vue' : 'vue';
    const filename = `${this.name}.${ext}`;
    const filepath = path.join(componentDir, filename);

    const template = this.vueTemplate();
    fs.writeFileSync(filepath, template);
    console.log(`✓ Created ${filename}`);
  }

  /**
   * Main component generation router
   */
  generateComponent(componentDir) {
    if (this.framework === 'react') {
      this.generateReactComponent(componentDir);
    } else if (this.framework === 'vue') {
      this.generateVueComponent(componentDir);
    } else {
      throw new Error(`Unsupported framework: ${this.framework}`);
    }
  }

  /**
   * Generate CSS module
   */
  generateStyles(componentDir) {
    const filename = `${this.name}.module.css`;
    const filepath = path.join(componentDir, filename);

    const template = `/* ${this.name} Styles */

.${this.name.toLowerCase()} {
  /* Base styles */
  display: block;
  position: relative;
}

.${this.name.toLowerCase()}--primary {
  /* Primary variant */
}

.${this.name.toLowerCase()}--secondary {
  /* Secondary variant */
}

.${this.name.toLowerCase()}--disabled {
  /* Disabled state */
  opacity: 0.5;
  cursor: not-allowed;
}

/* Responsive */
@media (max-width: 768px) {
  .${this.name.toLowerCase()} {
    /* Mobile styles */
  }
}
`;

    fs.writeFileSync(filepath, template);
    console.log(`✓ Created ${filename}`);
  }

  /**
   * Generate test file
   */
  generateTest(componentDir) {
    const ext = this.typescript ? 'tsx' : 'jsx';
    const filename = `${this.name}.test.${ext}`;
    const filepath = path.join(componentDir, filename);

    const template = this.framework === 'react'
      ? this.reactTestTemplate()
      : this.vueTestTemplate();

    fs.writeFileSync(filepath, template);
    console.log(`✓ Created ${filename}`);
  }

  /**
   * Generate Storybook story
   */
  generateStory(componentDir) {
    const ext = this.typescript ? 'tsx' : 'jsx';
    const filename = `${this.name}.stories.${ext}`;
    const filepath = path.join(componentDir, filename);

    const template = this.storybookTemplate();
    fs.writeFileSync(filepath, template);
    console.log(`✓ Created ${filename}`);
  }

  /**
   * Generate index file
   */
  generateIndex(componentDir) {
    const ext = this.typescript ? 'ts' : 'js';
    const filename = `index.${ext}`;
    const filepath = path.join(componentDir, filename);

    const template = `export { ${this.name} } from './${this.name}';\nexport type { ${this.name}Props } from './${this.name}';\n`;
    fs.writeFileSync(filepath, template);
    console.log(`✓ Created ${filename}`);
  }

  /**
   * React TypeScript template
   */
  reactTSTemplate() {
    return `import React from 'react';
import styles from './${this.name}.module.css';

export interface ${this.name}Props {
  variant?: 'primary' | 'secondary';
  disabled?: boolean;
  children?: React.ReactNode;
  className?: string;
  'aria-label'?: string;
  onClick?: () => void;
}

/**
 * ${this.name} component
 *
 * @example
 * <${this.name} variant="primary" onClick={handleClick}>
 *   Click me
 * </${this.name}>
 */
export const ${this.name}: React.FC<${this.name}Props> = ({
  variant = 'primary',
  disabled = false,
  children,
  className = '',
  'aria-label': ariaLabel,
  onClick,
}) => {
  const handleClick = () => {
    if (!disabled && onClick) {
      onClick();
    }
  };

  const classes = [
    styles.${this.name.toLowerCase()},
    styles[\`${this.name.toLowerCase()}--\${variant}\`],
    disabled && styles[\`${this.name.toLowerCase()}--disabled\`],
    className,
  ].filter(Boolean).join(' ');

  return (
    <div
      className={classes}
      role="region"
      aria-label={ariaLabel || '${this.name}'}
      aria-disabled={disabled}
      onClick={handleClick}
      onKeyDown={(e) => e.key === 'Enter' && handleClick()}
      tabIndex={disabled ? -1 : 0}
    >
      {children}
    </div>
  );
};

${this.name}.displayName = '${this.name}';
`;
  }

  /**
   * React JavaScript template
   */
  reactJSTemplate() {
    return `import React from 'react';
import styles from './${this.name}.module.css';

export const ${this.name} = ({
  variant = 'primary',
  disabled = false,
  children,
  className = '',
  'aria-label': ariaLabel,
  onClick,
}) => {
  const handleClick = () => {
    if (!disabled && onClick) {
      onClick();
    }
  };

  const classes = [
    styles.${this.name.toLowerCase()},
    styles[\`${this.name.toLowerCase()}--\${variant}\`],
    disabled && styles[\`${this.name.toLowerCase()}--disabled\`],
    className,
  ].filter(Boolean).join(' ');

  return (
    <div
      className={classes}
      role="region"
      aria-label={ariaLabel || '${this.name}'}
      aria-disabled={disabled}
      onClick={handleClick}
      onKeyDown={(e) => e.key === 'Enter' && handleClick()}
      tabIndex={disabled ? -1 : 0}
    >
      {children}
    </div>
  );
};

${this.name}.displayName = '${this.name}';
`;
  }

  /**
   * Vue template
   */
  vueTemplate() {
    const scriptLang = this.typescript ? 'lang="ts"' : '';
    return `<template>
  <div
    :class="classes"
    role="region"
    :aria-label="ariaLabel || '${this.name}'"
    :aria-disabled="disabled"
    :tabindex="disabled ? -1 : 0"
    @click="handleClick"
    @keydown.enter="handleClick"
  >
    <slot></slot>
  </div>
</template>

<script ${scriptLang}>
import { computed, defineComponent } from 'vue';

export default defineComponent({
  name: '${this.name}',
  props: {
    variant: {
      type: String,
      default: 'primary',
      validator: (value: string) => ['primary', 'secondary'].includes(value),
    },
    disabled: {
      type: Boolean,
      default: false,
    },
    ariaLabel: {
      type: String,
      default: '',
    },
  },
  emits: ['click'],
  setup(props, { emit }) {
    const handleClick = () => {
      if (!props.disabled) {
        emit('click');
      }
    };

    const classes = computed(() => [
      '${this.name.toLowerCase()}',
      \`${this.name.toLowerCase()}--\${props.variant}\`,
      props.disabled && '${this.name.toLowerCase()}--disabled',
    ].filter(Boolean).join(' '));

    return {
      handleClick,
      classes,
    };
  },
});
</script>

<style scoped>
@import './${this.name}.module.css';
</style>
`;
  }

  /**
   * React test template
   */
  reactTestTemplate() {
    return `import { render, screen, fireEvent } from '@testing-library/react';
import { ${this.name} } from './${this.name}';

describe('${this.name}', () => {
  it('renders children correctly', () => {
    render(<${this.name}>Test Content</${this.name}>);
    expect(screen.getByText('Test Content')).toBeInTheDocument();
  });

  it('applies primary variant by default', () => {
    const { container } = render(<${this.name}>Content</${this.name}>);
    expect(container.firstChild).toHaveClass('${this.name.toLowerCase()}--primary');
  });

  it('applies secondary variant when specified', () => {
    const { container } = render(<${this.name} variant="secondary">Content</${this.name}>);
    expect(container.firstChild).toHaveClass('${this.name.toLowerCase()}--secondary');
  });

  it('handles click events', () => {
    const handleClick = jest.fn();
    render(<${this.name} onClick={handleClick}>Click me</${this.name}>);

    fireEvent.click(screen.getByText('Click me'));
    expect(handleClick).toHaveBeenCalledTimes(1);
  });

  it('does not call onClick when disabled', () => {
    const handleClick = jest.fn();
    render(<${this.name} disabled onClick={handleClick}>Click me</${this.name}>);

    fireEvent.click(screen.getByText('Click me'));
    expect(handleClick).not.toHaveBeenCalled();
  });

  it('has correct accessibility attributes', () => {
    const { container } = render(
      <${this.name} aria-label="Custom label">Content</${this.name}>
    );

    const element = container.firstChild;
    expect(element).toHaveAttribute('role', 'region');
    expect(element).toHaveAttribute('aria-label', 'Custom label');
    expect(element).toHaveAttribute('tabindex', '0');
  });

  it('sets tabindex to -1 when disabled', () => {
    const { container } = render(<${this.name} disabled>Content</${this.name}>);
    expect(container.firstChild).toHaveAttribute('tabindex', '-1');
  });

  it('handles keyboard navigation (Enter key)', () => {
    const handleClick = jest.fn();
    render(<${this.name} onClick={handleClick}>Press me</${this.name}>);

    fireEvent.keyDown(screen.getByText('Press me'), { key: 'Enter' });
    expect(handleClick).toHaveBeenCalledTimes(1);
  });
});
`;
  }

  /**
   * Vue test template
   */
  vueTestTemplate() {
    return `import { mount } from '@vue/test-utils';
import ${this.name} from './${this.name}.vue';

describe('${this.name}', () => {
  it('renders slot content correctly', () => {
    const wrapper = mount(${this.name}, {
      slots: {
        default: 'Test Content',
      },
    });
    expect(wrapper.text()).toBe('Test Content');
  });

  it('applies primary variant by default', () => {
    const wrapper = mount(${this.name});
    expect(wrapper.classes()).toContain('${this.name.toLowerCase()}--primary');
  });

  it('applies secondary variant when specified', () => {
    const wrapper = mount(${this.name}, {
      props: {
        variant: 'secondary',
      },
    });
    expect(wrapper.classes()).toContain('${this.name.toLowerCase()}--secondary');
  });

  it('emits click event', async () => {
    const wrapper = mount(${this.name});
    await wrapper.trigger('click');
    expect(wrapper.emitted('click')).toHaveLength(1);
  });

  it('does not emit click when disabled', async () => {
    const wrapper = mount(${this.name}, {
      props: {
        disabled: true,
      },
    });
    await wrapper.trigger('click');
    expect(wrapper.emitted('click')).toBeUndefined();
  });

  it('has correct accessibility attributes', () => {
    const wrapper = mount(${this.name}, {
      props: {
        ariaLabel: 'Custom label',
      },
    });
    expect(wrapper.attributes('role')).toBe('region');
    expect(wrapper.attributes('aria-label')).toBe('Custom label');
    expect(wrapper.attributes('tabindex')).toBe('0');
  });
});
`;
  }

  /**
   * Storybook template
   */
  storybookTemplate() {
    return `import type { Meta, StoryObj } from '@storybook/react';
import { ${this.name} } from './${this.name}';

const meta: Meta<typeof ${this.name}> = {
  title: 'Components/${this.name}',
  component: ${this.name},
  tags: ['autodocs'],
  argTypes: {
    variant: {
      control: 'select',
      options: ['primary', 'secondary'],
    },
    disabled: {
      control: 'boolean',
    },
    onClick: { action: 'clicked' },
  },
};

export default meta;
type Story = StoryObj<typeof ${this.name}>;

export const Primary: Story = {
  args: {
    variant: 'primary',
    children: 'Primary ${this.name}',
  },
};

export const Secondary: Story = {
  args: {
    variant: 'secondary',
    children: 'Secondary ${this.name}',
  },
};

export const Disabled: Story = {
  args: {
    variant: 'primary',
    disabled: true,
    children: 'Disabled ${this.name}',
  },
};

export const WithCustomAriaLabel: Story = {
  args: {
    variant: 'primary',
    'aria-label': 'Custom accessible name',
    children: 'Accessible ${this.name}',
  },
};
`;
  }

  /**
   * Print usage instructions
   */
  printUsage() {
    console.log(`Usage:\n`);
    console.log(`  import { ${this.name} } from './components/${this.name}';\n`);
    console.log(`  <${this.name} variant="primary" onClick={handleClick}>`);
    console.log(`    Content`);
    console.log(`  </${this.name}>\n`);
  }
}

// CLI
if (require.main === module) {
  const args = process.argv.slice(2);
  const options = {};

  for (let i = 0; i < args.length; i += 2) {
    const key = args[i].replace('--', '');
    const value = args[i + 1];
    options[key] = value === 'true' ? true : value === 'false' ? false : value;
  }

  if (!options.name) {
    console.error('❌ Error: --name is required\n');
    console.log('Usage:');
    console.log('  node component-generator.js --framework react --name Button --type functional');
    console.log('  node component-generator.js --framework vue --name Card --type composition\n');
    process.exit(1);
  }

  try {
    const generator = new ComponentGenerator(options);
    generator.generate();
  } catch (error) {
    console.error(`❌ Error: ${error.message}`);
    process.exit(1);
  }
}

module.exports = { ComponentGenerator };
