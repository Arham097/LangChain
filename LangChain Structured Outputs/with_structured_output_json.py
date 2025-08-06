from typing import TypedDict, Annotated, Optional, Literal
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field
from dotenv import load_dotenv
load_dotenv()

json_schema = {
  "title": "ReviewAnalysis",
  "description": "Schema representing the structured output of a product review",
  "type": "object",
  "properties": {
    "key_themes": {
      "type": "array",
      "items": { "type": "string" },
      "description": "Write down all the key themes discussed in review in a list"
    },
    "summary": {
      "type": "string",
      "description": "A breif summary of the review"
    },
    "sentiment": {
      "type": "string",
      "enum": ["pos", "neg"],
      "description": "Return sentiment of the review either positive, negative or neutral"
    },
    "pros": {
      "type": ["array", "null"],
      "items": { "type": "string" },
      "description": "Write down all the pros inside a list"
    },
    "cons": {
      "type": ["array", "null"],
      "items": { "type": "string" },
      "description": "Write down all the cons inside a list"
    }
  },
  "required": ["key_themes", "summary", "sentiment"]
}


model = ChatGoogleGenerativeAI(model='gemini-1.5-flash')

structured_output = model.with_structured_output(json_schema)

result = structured_output.invoke("""
I recently upgraded to the Samsung Galaxy S24 Ultra, and I must say, it's an absolute powerhouse! The Snapdragon 8 Gen 3 processor makes everything lightning fast—whether I’m gaming, multitasking, or editing photos. The 5000mAh battery easily lasts a full day even with heavy use, and the 45W fast charging is a lifesaver.

The S-Pen integration is a great touch for note-taking and quick sketches, though I don’t use it often. What really blew me away is the 200MP camera—the night mode is stunning, capturing crisp, vibrant images even in low light. Zooming up to 100x actually works well for distant objects, but anything beyond 30x loses quality.

However, the weight and size make it a bit uncomfortable for one-handed use. Also, Samsung’s One UI still comes with bloatware—why do I need five different Samsung apps for things Google already provides? The $1,300 price tag is also a hard pill to swallow.

Pros:

Insanely powerful processor (great for gaming and productivity)
Stunning 200MP camera with incredible zoom capabilities
Long battery life with fast charging
S-Pen support is unique and useful

Cons:

Bulky and heavy—not great for one-handed use
Bloatware still exists in One UI
Expensive compared to competitors
""")

print(result)