# Chapter 5: Environment Configuration (.env)

In [Chapter 4: Tools](04_tools_.md), you learned how to give your agent **hands**—Python functions that let it accomplish real tasks in the world. Your agent can now call tools to look up information, perform calculations, and interact with other systems.

But here's a new problem: **Your agent needs secrets and configuration to run.**

Imagine you're building a weather agent that needs to call a real weather API. That API requires an API key—a secret password you need to keep safe. Or imagine your agent needs to know which cloud location to use (us-west1 vs. europe-west1?). These decisions need to be stored somewhere, and they change depending on whether you're testing locally or running in production.

If you hardcode these secrets directly into your agent code, you'll face serious problems:[1]

- **Security risk:** Someone sees your code on GitHub with your API key exposed 😱
- **Different environments:** Your laptop, testing server, and production all need different settings
- **Easy mistakes:** You might accidentally commit secrets to version control

**The `.env` file solves all of these problems.** It's a simple configuration file that stores all your secrets and settings in one organized place, separate from your code.[1][3][4]

Think of `.env` as your agent's **backstage control panel**—the place where you set up all the infrastructure your agent depends on without hardcoding values into the actual code.

## The Problem: Where Do Secrets Live?

Let's make this concrete with a real example.

**Without `.env` (the wrong way):**

```python
# agent.py
API_KEY = "sk-12345-secret-key-67890"  # ❌ EXPOSED!
WEATHER_API = "https://api.weather.com"
CLOUD_LOCATION = "us-west1"

my_agent = Agent(
    name="weather_bot",
    model="gemini-2.5-flash-lite"
)
```

**Problems:**
- Anyone who reads your code sees your secret API key
- If you push to GitHub, the secret is permanently visible in history
- Different environments (local, testing, production) all use the same values
- Changing settings means editing code and redeploying

**With `.env` (the right way):**

```python
# agent.py
import os
from dotenv import load_dotenv

load_dotenv()  # Load from .env file

API_KEY = os.environ.get("API_KEY")  # Safely read from .env
WEATHER_API = os.environ.get("WEATHER_API")
CLOUD_LOCATION = os.environ.get("CLOUD_LOCATION")

my_agent = Agent(
    name="weather_bot",
    model="gemini-2.5-flash-lite"
)
```

```python
# .env (gitignore this file - never commit it!)
API_KEY=sk-12345-secret-key-67890
WEATHER_API=https://api.weather.com
CLOUD_LOCATION=us-west1
```

**Benefits:**
- ✅ Secrets are separate from code
- ✅ Different environments can have different `.env` files
- ✅ Easy to change settings without editing code
- ✅ Can be safely ignored by version control (added to `.gitignore`)

## What is a `.env` File, Really?

A `.env` file is simply a **text file that stores key-value pairs of configuration settings.**[1][3][4]

Here's the simplest possible `.env` file:

```
DATABASE_URL=postgresql://user:pass@localhost/mydb
API_KEY=sk-12345-secret-key
DEBUG=true
```

**What this is:** Three configuration settings stored as key=value pairs. That's it! Nothing magical—just text.

Your agent reads this file and loads these values into memory. Then your code can access them safely:

```python
import os

# Access values from .env
db_url = os.environ.get("DATABASE_URL")
api_key = os.environ.get("API_KEY")
debug_mode = os.environ.get("DEBUG")
```

## The Three Key Concepts

Every `.env` file has three important ideas:

### 1. Key-Value Pairs (What Goes In?)

Each line has a key (the name) and a value (the setting):[1]

```
KEY=value
```

**Example:**
```
GOOGLE_CLOUD_LOCATION=global
GOOGLE_GENAI_USE_VERTEXAI=1
```

The key (`GOOGLE_CLOUD_LOCATION`) is like a variable name. The value (`global`) is what it's set to.

### 2. Environment Variables (How Your Code Reads It)

Your Python code reads from `.env` using `os.environ`:[1]

```python
import os

location = os.environ.get("GOOGLE_CLOUD_LOCATION")
# location = "global"
```

This is called an **environment variable**—a value that comes from the system environment, not hardcoded in code.

### 3. `.gitignore` (Keep Secrets Secret!)

You must tell Git to ignore your `.env` file so secrets don't get committed:[1]

```
# .gitignore
.env
.env.local
```

This ensures your secrets stay on your machine and never get uploaded to GitHub.

## A Real-World Example: Your Weather Agent's Setup

Let's see the complete setup from the provided notebook. Your agent deployment needs three configuration values:

```
# .env
GOOGLE_CLOUD_LOCATION="global"
GOOGLE_GENAI_USE_VERTEXAI=1
GOOGLE_CLOUD_PROJECT=my-project-id
```

**What each setting does:**

- `GOOGLE_CLOUD_LOCATION="global"` → Use Google's global endpoint for Gemini API
- `GOOGLE_GENAI_USE_VERTEXAI=1` → Use Vertex AI (not Google AI Studio)
- `GOOGLE_CLOUD_PROJECT=my-project-id` → Which Google Cloud project to use

Your agent code then reads these:

```python
import os

location = os.environ.get("GOOGLE_CLOUD_LOCATION")
use_vertex = os.environ.get("GOOGLE_GENAI_USE_VERTEXAI")
project = os.environ.get("GOOGLE_CLOUD_PROJECT")
```

**What happens:** These environment variables configure your agent to use the right cloud infrastructure without hardcoding it.

## How `.env` Files Work: The Journey

When your agent starts, here's what happens internally:

```mermaid
sequenceDiagram
    participant App as Your Agent
    participant DotEnv as dotenv Library
    participant File as .env File
    participant OS as Environment Variables
    
    App->>DotEnv: load_dotenv()
    DotEnv->>File: Read .env file
    File-->>DotEnv: Returns key-value pairs
    DotEnv->>OS: Set environment variables
    App->>OS: os.environ.get("KEY")
    OS-->>App: Returns value
```

**Step-by-step:**

1. **Your agent starts** → Python code runs

2. **Call load_dotenv()** → Load `.env` file into memory

3. **dotenv reads file** → Parse all key=value lines

4. **Set environment variables** → Add them to the system environment

5. **Your code reads them** → Access via `os.environ.get("KEY")`

6. **Agent uses values** → Configure infrastructure, set credentials, etc.

## Creating Your First `.env` File

Let's create a `.env` file for your project:[3][4]

### Step 1: Create the file

```bash
touch .env
```

Or manually create a blank file named `.env` in your project root.

### Step 2: Add configuration

Open `.env` in a text editor and add your settings:

```
# Weather API Configuration
WEATHER_API_KEY=sk-weather-12345
WEATHER_API_URL=https://api.weather.com

# Google Cloud Configuration
GOOGLE_CLOUD_LOCATION=global
GOOGLE_GENAI_USE_VERTEXAI=1
```

**Important:**
- One key=value pair per line
- No quotes needed (unless value has spaces)
- Comments start with `#`

### Step 3: Add to `.gitignore`

Make sure Git ignores this file:

```
# In your .gitignore file
.env
.env.local
```

This prevents secrets from being committed to version control.

## Using `.env` in Your Agent Code

Here's how to load and use `.env` values in your agent:[3]

```python
import os
from dotenv import load_dotenv

# Step 1: Load .env file
load_dotenv()

# Step 2: Read values
api_key = os.environ.get("WEATHER_API_KEY")
location = os.environ.get("GOOGLE_CLOUD_LOCATION")

# Step 3: Use them
print(f"Using API key: {api_key}")
print(f"Using location: {location}")
```

**What this does:**
- `load_dotenv()` reads your `.env` file and loads all values
- `os.environ.get()` safely reads a value (returns None if missing)
- Your code uses the values normally

## Different Environments: The Real Power

The real power of `.env` is that **different environments can have different settings.**

### Local Development `.env`

```
# .env (your laptop)
GOOGLE_CLOUD_LOCATION=us-west1
DEBUG=true
BATCH_SIZE=10
```

### Production `.env`

```
# .env (production server)
GOOGLE_CLOUD_LOCATION=europe-west1
DEBUG=false
BATCH_SIZE=1000
```

**Same code, different settings!** Your agent automatically adapts based on which `.env` file is loaded.

## Best Practices: Doing `.env` Right

### Practice 1: Use Descriptive Names

```python
# ✅ Good - clear what each value does
API_KEY=sk-123
DATABASE_URL=postgresql://...
BATCH_SIZE=100

# ❌ Bad - confusing names
KEY=sk-123
URL=postgresql://...
SIZE=100
```

### Practice 2: Group Related Settings

```python
# ✅ Good - organized by feature
WEATHER_API_KEY=sk-weather
WEATHER_API_URL=https://api.weather.com

SLACK_BOT_TOKEN=xoxb-slack
SLACK_CHANNEL=general
```

### Practice 3: Document Your `.env`

Create a `.env.example` file showing what variables are needed (without secrets):[1]

```
# .env.example (commit this to GitHub)
WEATHER_API_KEY=your_api_key_here
WEATHER_API_URL=https://api.weather.com
GOOGLE_CLOUD_LOCATION=global
DEBUG=true
```

Now developers know what to set up without exposing real secrets!

### Practice 4: Validate Required Settings

```python
import os
from dotenv import load_dotenv

load_dotenv()

# Check required variables
required = ["API_KEY", "GOOGLE_CLOUD_LOCATION"]
for var in required:
    if not os.environ.get(var):
        raise ValueError(f"Missing required variable: {var}")

print("✅ All required variables are set!")
```

This prevents your agent from starting with incomplete configuration.

## Common Questions Beginners Ask

**Q: Should I commit `.env` to GitHub?**
A: Never! Add it to `.gitignore` so it's never uploaded. Secrets should stay local.

**Q: What if I need different `.env` files for different environments?**
A: Create `.env.local` for local, `.env.production` for production. Load the right one based on your environment.

**Q: Can I use `.env` in production?**
A: Not recommended. Use your cloud provider's secret management instead (Google Secret Manager, AWS Secrets Manager, etc.). But for development, `.env` is perfect.

**Q: What if someone finds my `.env` file on my computer?**
A: The secrets are still compromised. Rotate them immediately. This is why production uses secret management tools.

## The `.env` File in Your Agent Project Structure

From the notebook, here's how `.env` fits into your project:[1]

```
sample_agent/
├── agent.py                    # Your agent code
├── requirements.txt            # Python dependencies
├── .env                        # Configuration (never commit!)
├── .agent_engine_config.json   # Deployment settings
└── .gitignore                  # Tells Git to ignore .env
```

The `.env` file sits at the root of your agent project. When you deploy with ADK, the `.env` file values are used to configure your agent's runtime environment.

## Summary: What You've Learned

The `.env` file is your **backstage control panel** for configuration:[1][3][4]

✅ **Store secrets safely** → Keep API keys, passwords out of code
✅ **Environment-specific settings** → Different values for local/production
✅ **Easy to change** → Update settings without touching code
✅ **Version control safe** → Add to `.gitignore`, never commit secrets
✅ **Simple format** → Just key=value pairs, easy to read

The beauty of `.env` is that it **separates configuration from code**. Your agent logic doesn't care where a setting comes from—it just reads from environment variables. This makes your agent flexible and secure.

---

**Next:** [Chapter 6: Requirements Management](06_requirements_management_.md) will teach you how to declare and manage the Python packages your agent depends on.

---

Generated by [Erwin R. Pasia](https://github.com/erwinpasia/Full-Stack-Data-Science)