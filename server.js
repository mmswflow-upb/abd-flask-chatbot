import OpenAI from "openai";
import express from "express";
import dotenv from "dotenv";
import cors from "cors";

dotenv.config();

const app = express();
const port = process.env.PORT || 3000;

// Enable CORS
app.use(cors());

// Middleware for parsing JSON
app.use(express.json());

const OPENAI_API_KEY = process.env.OPENAI_API_KEY;
const SECRET_KEY = process.env.SECRET_KEY;

// Initialize OpenAI
const openai = new OpenAI({
  apiKey: OPENAI_API_KEY,
});

// Store a single conversation in memory
let conversationHistory = [
  {
    role: "system",
    content:
      "You are a compassionate chatbot designed to support people diagnosed with depression. Respond with empathy and encouragement.",
  },
];

// Authorization function
function authorize(req) {
  const devKey = req.headers.devkey;
  if (devKey !== SECRET_KEY) {
    return false;
  }
  return true;
}

// Chat route
app.post("/chat", async (req, res) => {
  const { message } = req.body;

  if (!authorize(req)) {
    return res.status(401).json({ error: "Unauthorized" });
  }

  if (!message) {
    return res
      .status(400)
      .json({ error: "Request body must include a 'message' field." });
  }

  // Add the user's message to the conversation
  conversationHistory.push({ role: "user", content: message });

  try {
    // Send the conversation history to OpenAI
    const completion = await openai.chat.completions.create({
      model: "gpt-4o",
      messages: conversationHistory,
    });

    const reply = completion.choices[0].message.content;

    // Add the assistant's reply to the conversation
    conversationHistory.push({ role: "assistant", content: reply });

    // Truncate conversation history to avoid hitting token limits
    if (conversationHistory.length > 20) {
      conversationHistory = conversationHistory.slice(-20);
    }

    res.json({ reply });
  } catch (error) {
    console.error("Error:", error.response?.data || error.message);
    res.status(500).json({ error: "Something went wrong. Please try again." });
  }
});

// Route to clear conversation history
app.post("/clear", (req, res) => {
  if (!authorize(req)) {
    return res.status(401).json({ error: "Unauthorized" });
  }

  // Reset conversation history
  conversationHistory = [
    {
      role: "system",
      content:
        "You are a compassionate chatbot designed to support people diagnosed with depression. Respond with empathy and encouragement.",
    },
  ];

  res.json({ message: "Conversation history cleared." });
});

// Start the server
app.listen(port, () => {
  console.log(`Chatbot server is running on port ${port}`);
});
