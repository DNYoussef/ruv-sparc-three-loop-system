/**
 * React Component Template (TypeScript)
 *
 * A production-ready component template with:
 * - TypeScript strict typing
 * - Accessibility defaults (ARIA, semantic HTML)
 * - CSS module integration
 * - Props validation
 * - JSDoc documentation
 * - Display name for React DevTools
 *
 * @template ComponentName
 * @version 1.0.0
 */

import React from 'react';
import styles from './ComponentName.module.css';

export interface ComponentNameProps {
  /**
   * Visual variant of the component
   * @default 'primary'
   */
  variant?: 'primary' | 'secondary' | 'tertiary';

  /**
   * Size of the component
   * @default 'medium'
   */
  size?: 'small' | 'medium' | 'large';

  /**
   * Disabled state
   * @default false
   */
  disabled?: boolean;

  /**
   * Child elements
   */
  children?: React.ReactNode;

  /**
   * Additional CSS class names
   */
  className?: string;

  /**
   * Accessible label for screen readers
   */
  'aria-label'?: string;

  /**
   * Accessible description
   */
  'aria-describedby'?: string;

  /**
   * Click handler
   */
  onClick?: () => void;

  /**
   * Focus handler
   */
  onFocus?: () => void;

  /**
   * Blur handler
   */
  onBlur?: () => void;
}

/**
 * ComponentName component
 *
 * Detailed description of what this component does and when to use it.
 *
 * @example
 * ```tsx
 * <ComponentName variant="primary" size="medium">
 *   Content goes here
 * </ComponentName>
 * ```
 *
 * @example With custom styling
 * ```tsx
 * <ComponentName
 *   variant="secondary"
 *   className="custom-class"
 *   aria-label="Custom accessible name"
 * >
 *   Content
 * </ComponentName>
 * ```
 */
export const ComponentName: React.FC<ComponentNameProps> = ({
  variant = 'primary',
  size = 'medium',
  disabled = false,
  children,
  className = '',
  'aria-label': ariaLabel,
  'aria-describedby': ariaDescribedBy,
  onClick,
  onFocus,
  onBlur,
}) => {
  // Event handlers
  const handleClick = () => {
    if (!disabled && onClick) {
      onClick();
    }
  };

  const handleKeyDown = (event: React.KeyboardEvent<HTMLDivElement>) => {
    if (!disabled && (event.key === 'Enter' || event.key === ' ')) {
      event.preventDefault();
      onClick?.();
    }
  };

  // CSS classes
  const classes = [
    styles.componentname,
    styles[`componentname--${variant}`],
    styles[`componentname--${size}`],
    disabled && styles['componentname--disabled'],
    className,
  ]
    .filter(Boolean)
    .join(' ');

  // Accessibility attributes
  const ariaAttributes = {
    'aria-label': ariaLabel || 'ComponentName',
    'aria-describedby': ariaDescribedBy,
    'aria-disabled': disabled,
  };

  return (
    <div
      className={classes}
      role="region"
      tabIndex={disabled ? -1 : 0}
      onClick={handleClick}
      onKeyDown={handleKeyDown}
      onFocus={onFocus}
      onBlur={onBlur}
      {...ariaAttributes}
    >
      {children}
    </div>
  );
};

ComponentName.displayName = 'ComponentName';
