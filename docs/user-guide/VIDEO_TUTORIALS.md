# Video Tutorial Scripts & Recording Guide

This guide provides scripts and instructions for recording video tutorials for the rUv SPARC UI Dashboard.

## Table of Contents

1. [Recording Setup](#recording-setup)
2. [Tutorial 1: Getting Started](#tutorial-1-getting-started)
3. [Tutorial 2: Creating and Managing Tasks](#tutorial-2-creating-and-managing-tasks)
4. [Tutorial 3: Project Management](#tutorial-3-project-management)
5. [Tutorial 4: Agent Monitor](#tutorial-4-agent-monitor)
6. [Tutorial 5: API Integration](#tutorial-5-api-integration)
7. [Publishing Videos](#publishing-videos)

---

## Recording Setup

### Equipment & Software

**Required**:
- **Screen Recording**: OBS Studio (free)
  - Download: https://obsproject.com/
- **Microphone**: Built-in or USB mic (decent quality)
- **Screen Resolution**: 1920x1080 (Full HD)

**Optional**:
- **Webcam**: For picture-in-picture
- **Video Editing**: DaVinci Resolve (free) or Adobe Premiere
- **Audio Editing**: Audacity (free)

### OBS Studio Configuration

**1. Install OBS Studio**:
```bash
# Download from https://obsproject.com/
# Install with default settings
```

**2. Configure Settings**:

**Video Settings**:
- Settings → Video
- Base Resolution: 1920x1080
- Output Resolution: 1920x1080
- FPS: 30

**Audio Settings**:
- Settings → Audio
- Desktop Audio: Default
- Microphone: Your microphone
- Sample Rate: 48kHz

**Output Settings**:
- Settings → Output
- Output Mode: Simple
- Recording Quality: High Quality
- Recording Format: MP4
- Encoder: Software (x264)

**3. Create Scene**:

**"Screen Capture" Scene**:
1. Click "+" under Scenes → Name: "Screen Capture"
2. Click "+" under Sources → "Display Capture"
3. Select your monitor
4. Resize to fit canvas

**"Picture-in-Picture" Scene** (optional):
1. Add "Display Capture" (full screen)
2. Add "Video Capture Device" (webcam)
3. Resize webcam to 320x240 (bottom-right corner)
4. Add circular mask (optional)

**4. Hotkeys**:
- Settings → Hotkeys
- Start Recording: F9
- Stop Recording: F10
- Toggle Mute: F11

### Recording Best Practices

**Before Recording**:
1. Close unnecessary applications
2. Disable notifications (Windows: Focus Assist, macOS: Do Not Disturb)
3. Clear browser history (clean URLs)
4. Prepare demo data (tasks, projects)
5. Test microphone levels (speak normally, check VU meter)

**During Recording**:
1. Speak clearly and slowly
2. Avoid "um" and "uh" (pause instead)
3. Explain what you're doing as you do it
4. Move mouse slowly (viewers need to follow)
5. Zoom in for small UI elements (Ctrl++ to zoom)

**After Recording**:
1. Review footage (check audio sync)
2. Trim beginning/end (remove setup)
3. Add intro/outro (optional)
4. Export at 1080p 30fps
5. Upload to YouTube

---

## Tutorial 1: Getting Started

**Duration**: 3-4 minutes

**Target Audience**: Complete beginners

**Goals**:
- Install the dashboard
- Create first account
- Navigate the UI

### Script

**[0:00 - Introduction (15 seconds)]**

> Hi, I'm [Name], and in this tutorial, I'll show you how to get started with the rUv SPARC UI Dashboard. By the end of this video, you'll have the dashboard installed and running on your computer. Let's get started!

**[0:15 - Prerequisites (30 seconds)]**

> Before we begin, make sure you have three things installed:
>
> First, Docker Desktop. [Show Docker Desktop website]
>
> Second, Node.js version 18 or higher. [Show Node.js website]
>
> And third, Git for cloning the repository. [Show Git website]
>
> If you don't have these installed, pause the video and install them now. Links are in the description.

**[0:45 - Clone Repository (30 seconds)]**

> Now let's clone the repository. Open your terminal and run:
>
> [Type command slowly]
> ```
> git clone https://github.com/yourusername/ruv-sparc-ui-dashboard.git
> ```
>
> [Press Enter, wait for clone to complete]
>
> Great! Now navigate into the project:
>
> ```
> cd ruv-sparc-ui-dashboard
> ```

**[1:15 - Configure Environment (45 seconds)]**

> Next, we need to configure our environment variables. Copy the example file:
>
> ```
> cp .env.example .env
> ```
>
> [Open .env file in text editor]
>
> For now, we'll use the default settings. In production, you should change the JWT_SECRET and SESSION_SECRET to secure random values.
>
> [Scroll through .env file, highlighting important settings]
>
> The defaults are fine for local development. Save and close the file.

**[2:00 - Start Docker Services (30 seconds)]**

> Now let's start the database and cache services using Docker:
>
> ```
> docker-compose up -d
> ```
>
> [Wait for containers to start, show docker ps output]
>
> Perfect! PostgreSQL and Redis are now running. You can verify with:
>
> ```
> docker ps
> ```
>
> You should see two containers running.

**[2:30 - Install Dependencies (30 seconds)]**

> Now we'll install the backend and frontend dependencies.
>
> First, the backend:
>
> ```
> cd backend
> npm install
> ```
>
> [Wait for installation - speed up in editing]
>
> And the frontend:
>
> ```
> cd ../frontend
> npm install
> ```
>
> [Wait for installation - speed up in editing]

**[3:00 - Start Application (30 seconds)]**

> Now we're ready to start the application. Open two terminal windows.
>
> In the first terminal, start the backend:
>
> ```
> cd backend
> npm run dev
> ```
>
> [Show backend starting]
>
> In the second terminal, start the frontend:
>
> ```
> cd frontend
> npm start
> ```
>
> [Show frontend compiling and opening browser]

**[3:30 - Create Account & Wrap Up (30 seconds)]**

> The browser should open automatically to localhost:3000. Let's create an account.
>
> [Click "Create Account"]
>
> Enter your email, username, and password.
>
> [Fill in form]
>
> Click "Sign Up", and you're in!
>
> [Show dashboard]
>
> That's it! You've successfully installed the rUv SPARC UI Dashboard. In the next tutorial, we'll explore how to create and manage tasks. Thanks for watching!

**[4:00 - End]**

---

## Tutorial 2: Creating and Managing Tasks

**Duration**: 5-6 minutes

**Target Audience**: Users who completed Tutorial 1

**Goals**:
- Create tasks multiple ways
- Edit and reschedule tasks
- Use filters and search

### Script

**[0:00 - Introduction (15 seconds)]**

> Welcome back! In this tutorial, we'll learn how to create and manage tasks in the rUv SPARC UI Dashboard. We'll cover multiple ways to create tasks, how to edit them, and how to use filters and search. Let's dive in!

**[0:15 - Calendar Overview (30 seconds)]**

> This is the Calendar view, your central hub for task management.
>
> [Navigate to Calendar view]
>
> You can switch between Month, Week, Day, and Agenda views using these buttons.
>
> [Click through each view]
>
> For this tutorial, we'll use Month view. You can navigate between months using these arrows, or click "Today" to jump to the current date.

**[0:45 - Quick Create (45 seconds)]**

> The fastest way to create a task is by clicking on a date.
>
> [Click on a date in the calendar]
>
> A quick-add form appears. Enter a task title - let's say "Review project proposal".
>
> [Type task title]
>
> Press Enter, and the task is created with default settings.
>
> [Task appears on calendar]
>
> Notice the task appears on the calendar with a color indicating its priority. By default, tasks are medium priority.

**[1:30 - Full Task Form (1 minute)]**

> For more control, use the "New Task" button.
>
> [Click "+ New Task" button]
>
> This opens the full task form with all available options.
>
> [Show full form]
>
> Let's create a task called "Implement user authentication".
>
> [Fill in fields while explaining]
>
> - Title: "Implement user authentication"
> - Description: I'll add some details here... [type brief description]
> - Priority: Let's set this to "High"
> - Due Date: I'll schedule this for next week
> - Tags: "backend", "security"
> - Project: I'll link this to the "Website Redesign" project
>
> [Click "Create Task"]
>
> Great! The task now appears on the calendar.

**[2:30 - Drag-and-Drop Rescheduling (45 seconds)]**

> Rescheduling tasks is easy with drag-and-drop.
>
> [Click and hold on a task]
>
> Simply click and drag the task to a new date.
>
> [Drag task to different date]
>
> Notice the target date highlights in green to show it's valid. Release to reschedule.
>
> [Release mouse]
>
> The task updates immediately. You can also drag tasks between weeks in Week view to adjust both date and time.

**[3:15 - Editing Tasks (45 seconds)]**

> To edit a task, click on it to open the detail view.
>
> [Click on task]
>
> From here, you can see all task details, including description, tags, and comments.
>
> [Show detail panel]
>
> Click "Edit" to modify any field.
>
> [Click Edit button]
>
> Let's change the priority to "Critical" and add a note.
>
> [Make changes]
>
> Save your changes.
>
> [Click Save]
>
> The task updates, and notice the color changed to red, indicating critical priority.

**[4:00 - Filters and Search (1 minute)]**

> To find specific tasks, use the filters in the sidebar.
>
> [Show sidebar filters]
>
> You can filter by priority, status, project, or tags. Let's see all high-priority tasks.
>
> [Check "High" priority filter]
>
> The calendar now shows only high-priority tasks. You can combine filters - let's also filter by the "backend" tag.
>
> [Check "backend" tag]
>
> Now we see high-priority backend tasks only.
>
> [Clear filters]
>
> For text search, use the search bar at the top.
>
> [Click search bar]
>
> Search for "authentication".
>
> [Type and search]
>
> All tasks mentioning authentication are highlighted. You can also use advanced search with filters like "priority:high" or "tag:urgent".

**[5:00 - Completing and Deleting Tasks (30 seconds)]**

> To mark a task complete, click the checkbox.
>
> [Click checkbox on task]
>
> Completed tasks turn gray and can be hidden using the "Hide completed" filter.
>
> [Show filter]
>
> To delete a task, right-click and select "Delete".
>
> [Right-click, select Delete]
>
> Confirm the deletion.
>
> [Confirm]
>
> The task is removed from the calendar.

**[5:30 - Wrap Up (15 seconds)]**

> That's it! You now know how to create, edit, reschedule, and find tasks. In the next tutorial, we'll explore project management with Kanban boards and analytics. Thanks for watching!

**[5:45 - End]**

---

## Tutorial 3: Project Management

**Duration**: 4-5 minutes

**Target Audience**: Users familiar with basic task management

**Goals**:
- Create and configure projects
- Use Kanban board view
- View project analytics

### Script

**[0:00 - Introduction (15 seconds)]**

> In this tutorial, we'll learn how to manage projects in the rUv SPARC UI Dashboard. We'll create a project, organize tasks on a Kanban board, and explore project analytics. Let's get started!

**[0:15 - Create Project (45 seconds)]**

> To create a project, click the "Projects" tab, then "New Project".
>
> [Navigate to Projects, click "+ New Project"]
>
> Let's create a project called "Mobile App Redesign".
>
> [Fill in form]
>
> - Name: "Mobile App Redesign"
> - Description: "Complete redesign of our mobile application"
> - Color: I'll choose blue
> - Icon: Let's use the mobile phone emoji 📱
> - Tags: "mobile", "design"
>
> [Click "Create Project"]
>
> Perfect! Our project is created. Now let's add some tasks.

**[1:00 - Add Tasks to Project (30 seconds)]**

> From the project view, click "Add Task".
>
> [Click "+ Add Task"]
>
> Let's add a few tasks:
> - "Design new UI mockups"
> - "Implement navigation redesign"
> - "User testing"
>
> [Quickly create 3-4 tasks]
>
> These tasks automatically link to our project.

**[1:30 - Kanban Board View (1 minute 30 seconds)]**

> Now let's switch to Board view to see our tasks as a Kanban board.
>
> [Click "Board" view]
>
> The board has four default columns: Backlog, In Progress, Review, and Done.
>
> [Show columns]
>
> To move a task through the workflow, simply drag it to the next column.
>
> [Drag task from Backlog to In Progress]
>
> Let's say we've finished the UI mockups and they're ready for review.
>
> [Drag task to Review column]
>
> You can customize columns by clicking "Customize Board".
>
> [Click "Customize Board"]
>
> Add, remove, or rename columns to match your workflow. You can also set WIP (work-in-progress) limits to prevent overloading a column.
>
> [Show customization options]
>
> For now, we'll keep the defaults.

**[3:00 - Timeline View (45 seconds)]**

> The Timeline view shows tasks as a Gantt chart.
>
> [Switch to Timeline view]
>
> Each task is represented as a horizontal bar showing its duration and due date.
>
> [Show timeline]
>
> You can drag tasks to reschedule them, or resize to adjust duration. Let's create a dependency - the navigation redesign depends on the mockups being complete.
>
> [Click and drag from mockups task to navigation task]
>
> Now the timeline shows that navigation can't start until mockups are done. This helps visualize project flow and identify bottlenecks.

**[3:45 - Project Analytics (45 seconds)]**

> Finally, let's look at project analytics. Click the "Analytics" tab.
>
> [Click "Analytics" tab]
>
> Here you see key metrics: total tasks, completion percentage, and progress over time.
>
> [Show metrics dashboard]
>
> The burndown chart shows how much work remains. The ideal line represents perfect progress, while the actual line shows real progress.
>
> [Point to burndown chart]
>
> Velocity shows tasks completed per week, helping you predict when the project will finish.
>
> [Show velocity chart]
>
> You can also generate reports for stakeholders.

**[4:30 - Wrap Up (15 seconds)]**

> That's project management in the rUv SPARC UI Dashboard! You can now organize tasks, track progress, and generate insights. In the next tutorial, we'll explore the Agent Monitor for workflow automation. Thanks for watching!

**[4:45 - End]**

---

## Tutorial 4: Agent Monitor

**Duration**: 4-5 minutes

**Target Audience**: Users interested in automation

**Goals**:
- Understand AI agents
- Monitor agent activity
- Trigger workflows

### Script

**[0:00 - Introduction (15 seconds)]**

> Welcome to the Agent Monitor tutorial! In this video, we'll explore how AI agents automate tasks, how to monitor their activity, and how to trigger custom workflows. Let's dive into the future of task automation!

**[0:15 - What are Agents? (45 seconds)]**

> Agents are autonomous AI workers that execute tasks based on workflows.
>
> [Navigate to Agent Monitor]
>
> The rUv SPARC system uses specialized agents for different tasks:
>
> [Show agent list]
>
> - The Planner breaks projects into tasks
> - The Coder generates code implementations
> - The Tester creates and runs tests
> - The Reviewer checks code quality
> - And several others for architecture, analysis, and documentation
>
> Each agent works independently but coordinates through a shared memory system. Let's see them in action.

**[1:00 - Agent Activity Dashboard (1 minute)]**

> This is the Agent Activity Dashboard, showing real-time status of all agents.
>
> [Show activity feed]
>
> Right now, the Coder agent is working on implementing an authentication feature. You can see:
>
> - The agent name and type
> - Current task
> - Status (Working, Idle, Waiting, or Error)
> - Elapsed time
> - Progress bar
>
> [Point to each element]
>
> The status grid shows all agents at a glance.
>
> [Show grid view]
>
> Idle agents are available for new tasks. Working agents show their current assignment. You can click any agent to see detailed logs.

**[2:00 - Workflow Visualization (1 minute)]**

> The workflow graph visualizes how agents coordinate.
>
> [Show workflow graph]
>
> Nodes represent agents, and edges show task flow. For example, this feature development workflow:
>
> [Trace through workflow]
>
> 1. Planner creates the task breakdown
> 2. Coder implements the feature
> 3. Tester writes and runs tests
> 4. Reviewer checks code quality
>
> The graph is interactive - hover over a node to see details.
>
> [Hover over nodes]
>
> Green nodes are complete, blue are in progress, and gray are waiting. If an agent encounters an error, the node turns red.

**[3:00 - Agent Logs (45 seconds)]**

> To debug issues or understand what an agent did, check the logs.
>
> [Click on an agent to open logs panel]
>
> Logs show chronological entries with timestamps and levels:
>
> [Scroll through logs]
>
> - INFO messages for general progress
> - DEBUG for detailed diagnostics
> - WARNING for non-critical issues
> - ERROR for failures
> - SUCCESS for completions
>
> You can filter by level, search for keywords, or export logs for offline analysis.
>
> [Show filter options]

**[3:45 - Triggering Workflows (45 seconds)]**

> To manually trigger a workflow, click "Trigger Workflow".
>
> [Click "Trigger Workflow" button]
>
> Select a template - let's use "Feature Development".
>
> [Select template]
>
> Configure parameters like project name and feature description.
>
> [Fill in parameters]
>
> Click "Start Workflow", and the agents spring into action!
>
> [Start workflow, show agents activating]
>
> You can monitor progress in real-time and receive notifications when each stage completes.

**[4:30 - Wrap Up (15 seconds)]**

> That's the Agent Monitor! You now understand how AI agents work together to automate complex tasks. In the final tutorial, we'll cover API integration for programmatic access. Thanks for watching!

**[4:45 - End]**

---

## Tutorial 5: API Integration

**Duration**: 5-6 minutes

**Target Audience**: Developers

**Goals**:
- Authenticate with API
- Create tasks via API
- Use WebSocket for real-time updates

### Script

**[0:00 - Introduction (15 seconds)]**

> In this tutorial, we'll explore the rUv SPARC API for programmatic access. We'll authenticate, create tasks, and subscribe to real-time updates. Whether you're building integrations or automating workflows, this is for you!

**[0:15 - Authentication (1 minute)]**

> First, we need an API token. There are two ways to authenticate: JWT tokens or API keys.
>
> [Show terminal]
>
> For JWT, we'll use the login endpoint:
>
> ```bash
> curl -X POST http://localhost:3001/api/v1/auth/login \
>   -H "Content-Type: application/json" \
>   -d '{"email": "admin@example.com", "password": "your-password"}'
> ```
>
> [Run command, show response]
>
> The response includes a token. Copy this token - we'll use it for subsequent requests.
>
> [Copy token]
>
> Alternatively, generate an API key from the web UI: Settings → API Access → Generate Key.
>
> [Show in browser]
>
> API keys don't expire, making them ideal for long-running integrations.

**[1:15 - Creating Tasks via API (1 minute 30 seconds)]**

> Now let's create a task using our token.
>
> [Show terminal]
>
> ```bash
> curl -X POST http://localhost:3001/api/v1/tasks \
>   -H "Authorization: Bearer YOUR_TOKEN" \
>   -H "Content-Type: application/json" \
>   -d '{
>     "title": "Task created via API",
>     "description": "This task was created programmatically",
>     "priority": "high",
>     "due_date": "2025-01-15T14:00:00Z"
>   }'
> ```
>
> [Run command]
>
> Success! The response includes the created task with its ID.
>
> [Show response]
>
> Let's verify in the UI.
>
> [Switch to browser, refresh calendar]
>
> There it is! The task appears in the calendar exactly as we specified.
>
> You can also update tasks with PATCH:
>
> ```bash
> curl -X PATCH http://localhost:3001/api/v1/tasks/TASK_ID \
>   -H "Authorization: Bearer YOUR_TOKEN" \
>   -H "Content-Type: application/json" \
>   -d '{"status": "completed"}'
> ```

**[2:45 - Listing and Filtering (1 minute)]**

> To retrieve tasks, use the GET endpoint with query parameters:
>
> ```bash
> curl "http://localhost:3001/api/v1/tasks?priority=high&status=pending&limit=10" \
>   -H "Authorization: Bearer YOUR_TOKEN"
> ```
>
> [Run command, show response]
>
> The response includes:
> - An array of tasks matching the filters
> - Pagination metadata
>
> [Show JSON structure]
>
> You can filter by priority, status, project, tags, due dates, and more. Sort results and paginate through large datasets.
>
> For a specific task, use its ID:
>
> ```bash
> curl http://localhost:3001/api/v1/tasks/TASK_ID \
>   -H "Authorization: Bearer YOUR_TOKEN"
> ```

**[3:45 - Real-time Updates with WebSockets (1 minute 30 seconds)]**

> For real-time updates, connect to the WebSocket server.
>
> [Show Node.js code in editor]
>
> Here's a simple example:
>
> ```javascript
> const WebSocket = require('ws');
>
> const ws = new WebSocket('ws://localhost:3002');
>
> ws.on('open', () => {
>   // Authenticate
>   ws.send(JSON.stringify({
>     type: 'auth',
>     token: 'YOUR_TOKEN'
>   }));
> });
>
> ws.on('message', (data) => {
>   const message = JSON.parse(data);
>
>   if (message.type === 'task_created') {
>     console.log('New task:', message.data);
>   }
> });
> ```
>
> [Run the code]
>
> Now when a task is created - either via UI or API - we receive a real-time notification.
>
> [Create task in UI, show WebSocket message in terminal]
>
> Perfect! You can subscribe to different channels: tasks, projects, agents, or workflows.

**[5:15 - Code Examples (30 seconds)]**

> Full code examples are available in multiple languages:
>
> [Show documentation in browser]
>
> - JavaScript (Node.js with Axios)
> - Python (requests library)
> - cURL (for shell scripts)
>
> Each example includes authentication, CRUD operations, error handling, and pagination. Check the API Guide in the documentation.

**[5:45 - Wrap Up (15 seconds)]**

> That's it! You now know how to integrate the rUv SPARC API into your applications. Build custom tools, automate workflows, or integrate with external services. Happy coding, and thanks for watching!

**[6:00 - End]**

---

## Publishing Videos

### YouTube Upload Checklist

**Before Upload**:
- [ ] Video exported at 1080p 30fps
- [ ] Audio levels normalized (-14 LUFS)
- [ ] Intro/outro added (optional)
- [ ] Captions/subtitles generated (auto or manual)

**Video Details**:

**Title Format**:
```
rUv SPARC UI Dashboard Tutorial #X: [Topic]
```

Examples:
- rUv SPARC UI Dashboard Tutorial #1: Getting Started
- rUv SPARC UI Dashboard Tutorial #2: Creating and Managing Tasks

**Description Template**:
```
In this tutorial, we'll [brief summary].

⏱️ Timestamps:
0:00 - Introduction
0:15 - [Section 1]
1:00 - [Section 2]
...

🔗 Links:
- Documentation: https://github.com/yourusername/ruv-sparc-ui-dashboard
- Installation Guide: [link]
- Discord Community: https://discord.gg/ruv-sparc

📚 Related Tutorials:
- Tutorial #1: Getting Started
- Tutorial #2: Creating Tasks

---

The rUv SPARC UI Dashboard is an open-source task management and workflow automation platform. Learn more at [link].

#ruvSPARC #TaskManagement #Productivity #OpenSource #Tutorial
```

**Tags**:
```
ruv sparc, task management, productivity, open source, tutorial, project management, AI agents, workflow automation, calendar, kanban board, API, WebSocket
```

**Thumbnail**:
- 1280x720 resolution
- Clear title text (large font)
- Screenshot of relevant UI
- Consistent branding across series

**Playlist**:
- Create playlist: "rUv SPARC UI Dashboard Tutorials"
- Add all tutorials in order

### Embedding in Documentation

**Update USER_GUIDE.md**:

```markdown
## Video Tutorials

### Getting Started (3:45)
[![Getting Started](https://img.youtube.com/vi/VIDEO_ID/0.jpg)](https://www.youtube.com/watch?v=VIDEO_ID)

Learn how to install and set up the rUv SPARC UI Dashboard.

### Creating and Managing Tasks (5:30)
[![Creating Tasks](https://img.youtube.com/vi/VIDEO_ID/0.jpg)](https://www.youtube.com/watch?v=VIDEO_ID)

Master task creation, editing, and organization.

[... more videos ...]
```

### Hosting Options

**Primary**: YouTube (recommended)
- Free, unlimited storage
- Good SEO and discoverability
- Embed support
- Analytics

**Alternative**: Vimeo
- Ad-free playback
- Professional appearance
- Privacy controls
- Paid plans for higher limits

**Self-hosted**: Not recommended
- Bandwidth costs
- Storage costs
- No SEO benefits
- Requires CDN

---

## Maintenance & Updates

**When to Update Videos**:
- Major UI changes
- New features
- Critical bug fixes
- User feedback (confusing sections)

**Versioning**:
- Add version number to description
- Pin comment with update notes
- Link to updated tutorial (if applicable)

**Community Contributions**:
- Accept community-submitted tutorials
- Feature in "Community Tutorials" playlist
- Credit creators in description

---

**Recording complete?** Upload to YouTube and embed in documentation! 🎥✨
