<!--
  Vue Component Template (Composition API + TypeScript)

  A production-ready Vue 3 component template with:
  - Composition API setup
  - TypeScript strict typing
  - Accessibility defaults (ARIA, semantic HTML)
  - Scoped styles
  - Props validation
  - Emits definition
  - Computed properties
  - JSDoc documentation

  @template ComponentName
  @version 1.0.0
-->

<template>
  <div
    :class="classes"
    role="region"
    :aria-label="ariaLabel || 'ComponentName'"
    :aria-describedby="ariaDescribedBy"
    :aria-disabled="disabled"
    :tabindex="disabled ? -1 : 0"
    @click="handleClick"
    @keydown.enter="handleClick"
    @keydown.space.prevent="handleClick"
    @focus="handleFocus"
    @blur="handleBlur"
  >
    <slot></slot>
  </div>
</template>

<script lang="ts">
import { computed, defineComponent, PropType } from 'vue';

export type Variant = 'primary' | 'secondary' | 'tertiary';
export type Size = 'small' | 'medium' | 'large';

/**
 * ComponentName component
 *
 * Detailed description of what this component does and when to use it.
 *
 * @example
 * ```vue
 * <ComponentName variant="primary" size="medium">
 *   Content goes here
 * </ComponentName>
 * ```
 */
export default defineComponent({
  name: 'ComponentName',

  props: {
    /**
     * Visual variant of the component
     * @default 'primary'
     */
    variant: {
      type: String as PropType<Variant>,
      default: 'primary',
      validator: (value: string): boolean =>
        ['primary', 'secondary', 'tertiary'].includes(value),
    },

    /**
     * Size of the component
     * @default 'medium'
     */
    size: {
      type: String as PropType<Size>,
      default: 'medium',
      validator: (value: string): boolean =>
        ['small', 'medium', 'large'].includes(value),
    },

    /**
     * Disabled state
     * @default false
     */
    disabled: {
      type: Boolean,
      default: false,
    },

    /**
     * Accessible label for screen readers
     */
    ariaLabel: {
      type: String,
      default: '',
    },

    /**
     * Accessible description
     */
    ariaDescribedBy: {
      type: String,
      default: '',
    },
  },

  emits: {
    /**
     * Emitted when component is clicked
     */
    click: null,

    /**
     * Emitted when component receives focus
     */
    focus: null,

    /**
     * Emitted when component loses focus
     */
    blur: null,
  },

  setup(props, { emit }) {
    // Computed properties
    const classes = computed(() =>
      [
        'componentname',
        `componentname--${props.variant}`,
        `componentname--${props.size}`,
        props.disabled && 'componentname--disabled',
      ]
        .filter(Boolean)
        .join(' ')
    );

    // Event handlers
    const handleClick = () => {
      if (!props.disabled) {
        emit('click');
      }
    };

    const handleFocus = () => {
      if (!props.disabled) {
        emit('focus');
      }
    };

    const handleBlur = () => {
      if (!props.disabled) {
        emit('blur');
      }
    };

    return {
      classes,
      handleClick,
      handleFocus,
      handleBlur,
    };
  },
});
</script>

<style scoped>
/* ComponentName Styles */

.componentname {
  /* Base styles */
  display: block;
  position: relative;
  cursor: pointer;
  transition: all 0.2s ease-in-out;
}

.componentname:focus {
  outline: 2px solid var(--focus-color, #0066cc);
  outline-offset: 2px;
}

/* Variants */
.componentname--primary {
  background-color: var(--primary-bg, #0066cc);
  color: var(--primary-text, #ffffff);
}

.componentname--secondary {
  background-color: var(--secondary-bg, #6c757d);
  color: var(--secondary-text, #ffffff);
}

.componentname--tertiary {
  background-color: var(--tertiary-bg, #f8f9fa);
  color: var(--tertiary-text, #212529);
}

/* Sizes */
.componentname--small {
  padding: 0.5rem 1rem;
  font-size: 0.875rem;
}

.componentname--medium {
  padding: 0.75rem 1.5rem;
  font-size: 1rem;
}

.componentname--large {
  padding: 1rem 2rem;
  font-size: 1.25rem;
}

/* Disabled state */
.componentname--disabled {
  opacity: 0.5;
  cursor: not-allowed;
  pointer-events: none;
}

/* Responsive */
@media (max-width: 768px) {
  .componentname--large {
    padding: 0.75rem 1.5rem;
    font-size: 1.125rem;
  }
}
</style>
