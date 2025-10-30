---
skill: gemini-media
description: Generate images and videos using Gemini's Imagen and Veo integration
tags: [gemini, image-generation, video-generation, visualization, media, imagen, veo]
version: 1.0.0
---

# Gemini Media Generation Skill

## Purpose
Leverage Gemini CLI's integration with Imagen (image generation) and Veo (video generation) to create visual content, diagrams, UI mockups, and demo videos - capabilities Claude Code cannot provide.

## Unique Capability
**What Claude Code Can't Do**: Generate images or videos. Gemini CLI can invoke Google's Imagen 3/4 for image generation and Veo 2/3.1 for video generation directly from the terminal, enabling automated visual content creation.

## When to Use

### Perfect For:
✅ Creating architectural diagrams and flowcharts
✅ Generating UI mockups and wireframes
✅ Producing demo videos for features
✅ Visualizing data structures and algorithms
✅ Creating documentation images
✅ Generating placeholder images for prototypes
✅ Making tutorial videos or walkthroughs
✅ Producing marketing/demo content

### Don't Use When:
❌ Need photo editing (not generation)
❌ Working with existing images (Gemini analyzes, but Claude Code can too)
❌ Need precision CAD or technical drawings
❌ Require specific brand assets (use actual assets)

## How It Works

This skill spawns a **Gemini Media Agent** that:
1. Uses Gemini CLI with MCP server for Imagen/Veo access
2. Generates images or videos based on text descriptions
3. Saves generated media to specified location
4. Returns file path and preview to Claude Code

## Usage

### Basic Image Generation
```
/gemini-media "Create a flowchart showing user authentication flow"
```

### Specific Image Request
```
/gemini-media "Generate a wireframe mockup for a dashboard with sidebar navigation, data table, and charts"
```

### Video Generation
```
/gemini-media "Create a short video showing a ginger cat exploring Australia" --type video
```

## Input Examples

### Diagrams & Flowcharts
```bash
# Architecture diagram
/gemini-media "Create an architecture diagram showing microservices: API Gateway, Auth Service, User Service, Database with arrows showing data flow"

# Flowchart
/gemini-media "Generate a flowchart for password reset process: user requests reset → email sent → click link → enter new password → success"

# State diagram
/gemini-media "Create a state diagram for order processing: pending → processing → shipped → delivered"
```

### UI Mockups
```bash
# Dashboard
/gemini-media "Design a modern dashboard UI with dark theme: top nav bar, left sidebar with icons, main area with cards showing metrics"

# Login page
/gemini-media "Create a clean login page mockup: centered card, email/password fields, login button, forgot password link"

# Mobile app screen
/gemini-media "Generate a mobile app screen for task management: header with '+' button, list of tasks with checkboxes, bottom navigation"
```

### Documentation Images
```bash
# Concept illustration
/gemini-media "Illustrate the concept of event-driven architecture with colorful icons and arrows"

# Before/after comparison
/gemini-media "Create a before/after comparison image showing code refactoring improvement"

# Technology stack visualization
/gemini-media "Visualize a modern web stack: React frontend, Node.js backend, PostgreSQL database, Redis cache"
```

### Videos (with --type video flag)
```bash
# Feature demo
/gemini-media "Create a 10-second video demonstrating a user clicking through an app onboarding flow" --type video

# Tutorial
/gemini-media "Generate a short video showing how to use a command line tool with text overlays" --type video

# Concept animation
/gemini-media "Animate the flow of data through a CI/CD pipeline from commit to deployment" --type video
```

## Output

The agent provides:
- **File Path**: Where the generated media was saved
- **Preview**: Description of what was generated
- **Specifications**: Resolution, format, file size
- **Generation Details**: Model used, parameters
- **Suggestions**: Potential refinements or variations

## Real-World Examples

### Example 1: Architecture Diagram
```
Request: "Create a microservices architecture diagram"

Agent generates:
- PNG image (1024x1024)
- Shows: API Gateway, 4 services, databases, message queue
- Arrows indicating data flow
- Color-coded by service type
- Saved to: docs/images/architecture.png
```

### Example 2: UI Mockup
```
Request: "Design an admin dashboard"

Agent generates:
- PNG mockup (1920x1080)
- Shows: Sidebar nav, header, metrics cards, charts, data table
- Modern design with consistent colors
- Saved to: designs/admin-dashboard-v1.png
```

### Example 3: Demo Video
```
Request: "Show user completing a checkout flow" --type video

Agent generates:
- MP4 video (10 seconds, 1080p)
- Shows: Cart → Shipping → Payment → Confirmation
- Smooth transitions between steps
- Saved to: demos/checkout-flow-demo.mp4
```

## Technical Details

### Gemini CLI Command Pattern

#### Image Generation (Imagen)
```bash
# Via MCP server
gemini "Generate an image: [detailed description]. Save to [path]"

# Using Imagen directly via API in script
# (Agent handles this internally)
```

#### Video Generation (Veo)
```bash
# Via MCP server
gemini "Generate a video: [detailed description]. Duration: [seconds]. Save to [path]"
```

### Available Models
- **Imagen 3**: High-quality images, better prompt adherence
- **Imagen 4**: Latest model, improved quality and control
- **Veo 2**: Video generation, general availability
- **Veo 3.1**: Enhanced audio, narrative control, new features
- **Veo 3.1 Fast**: Faster generation, paid preview

### Generation Parameters
- **Image Resolution**: 1024x1024 (default), custom sizes available
- **Video Duration**: Typically 10-30 seconds
- **Format**: PNG for images, MP4 for videos
- **Quality**: Adjustable based on use case

## Best Practices

### Write Detailed Descriptions
✅ "Modern dashboard with dark theme, sidebar navigation with icons, main area showing 4 metric cards in a grid, line chart below, data table at bottom"
❌ "Dashboard" (too vague)

### Specify Style and Details
✅ "Flowchart with rounded rectangles, blue for processes, green for decisions, orange for endpoints, arrows showing flow"
❌ "Flowchart" (unclear style)

### Include Layout Information
✅ "Mobile UI mockup, portrait orientation, header at top with back button, content area with list items, bottom navigation with 4 icons"
❌ "Mobile UI" (missing structure)

### For Videos, Describe Sequence
✅ "Video showing: 1) User lands on homepage 2) Clicks signup 3) Fills form 4) Receives confirmation. Each step 3 seconds."
❌ "Signup video" (no sequence)

## Integration with Workflow

### Typical Workflow:
1. Use `/gemini-media` to generate visual assets
2. Save to appropriate docs/images/designs directory
3. Reference in documentation or presentations
4. Use Claude Code to integrate into codebase if needed

### Works Well With:
- **Documentation generation**: Create diagrams for docs
- **gemini-megacontext**: Analyze codebase, then visualize architecture
- **Prototyping**: Generate UI mockups before implementation
- **Tutorials**: Create step-by-step visual guides

## Cost Considerations

- **Free tier**: Limited generation (check current quotas)
- **Paid tier**: More generations, higher quality, faster processing
- **API costs**: Imagen and Veo have separate pricing
- **Best practice**: Use for final assets, not rapid iteration

## Troubleshooting

### Generation Failed
→ Check MCP server is configured for Imagen/Veo access
→ Verify API credentials are set up
→ Ensure prompt is clear and specific

### Poor Quality Results
→ Provide more detailed descriptions
→ Specify style, colors, layout explicitly
→ Try different prompts with more context

### Wrong Style/Content
→ Be more prescriptive about visual style
→ Include examples or reference styles
→ Iterate with refinements

### File Not Saved
→ Check file path is valid and writable
→ Verify agent has permissions to create files
→ Ensure directory exists

## Limitations

⚠️ **Generation Limitations**:
- Cannot edit existing images (only generate new)
- Limited control over exact positioning
- May not match brand guidelines perfectly
- Video length typically limited to 30 seconds

⚠️ **Quality Considerations**:
- Generated images are AI-created (not designer-crafted)
- May need refinement for production use
- Cannot guarantee exact reproduction of complex layouts

✅ **Best For**:
- Rapid prototyping and ideation
- Documentation and diagrams
- Placeholder content
- Concept visualization
- Demo and presentation materials

## Advanced Features

### Image-to-Video
```bash
/gemini-media "Take this screenshot and animate it showing user clicking through the form" --input screenshot.png --type video
```

### Iterative Refinement
```bash
/gemini-media "Create dashboard mockup"
# Review output
/gemini-media "Refine previous dashboard: make sidebar wider, add profile icon top-right, change color scheme to blue"
```

### Batch Generation
```bash
/gemini-media "Generate 5 variations of logo design with different color schemes"
```

## Related Skills

- `gemini-megacontext`: Analyze architecture → visualize with media
- `gemini-extensions`: Use Figma extension → enhance with generated assets
- `multi-model`: Let orchestrator decide when visualization needed

## Success Indicators

✅ Generated image/video matches description
✅ Quality suitable for intended use
✅ File saved to correct location
✅ Asset integrated into documentation/project
✅ Time saved vs manual creation
✅ Visualization enhances understanding

---

**Remember**: Use Gemini Media for GENERATION (creating new visual assets), not EDITING (modifying existing ones). Best for rapid prototyping and documentation.
