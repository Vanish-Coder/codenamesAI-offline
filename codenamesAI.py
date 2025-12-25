#This is not necessarily AI in the sense of ChatGPT/Gemini. Rather it uses Sentence Transformers (https://sbert.net/), which creates mathemtical vector representations
#of all the words. It then runs a bunch of mathmetical calclations determiend by a LLM (specifically a NLP) and returns a bunch of best matches for the specific usecase.

import json
import numpy as np
from sentence_transformers import SentenceTransformer
import random

# Set seed based on current time for variation
np.random.seed(None)
random.seed(None)

# -----------------------------
# 1. Load the AI model
# -----------------------------
model = SentenceTransformer("all-MiniLM-L6-v2")  # free, fast

# -----------------------------
# 2. Read the board from state.json
# -----------------------------
with open("state.json", "r") as f:
    board = json.load(f)

red_words = [w.upper() for w in board.get("red_words", [])]
blue_words = [w.upper() for w in board.get("blue_words", [])]
neutral_words = [w.upper() for w in board.get("neutral_words", [])]
assassin = board.get("assassin", "").upper()
revealed_words = set([w.upper() for w in board.get("revealed", [])])

# Filter out already revealed words from red_words
red_words = [w for w in red_words if w not in revealed_words]
blue_words = [w for w in blue_words if w not in revealed_words]
neutral_words = [w for w in neutral_words if w not in revealed_words]

# If assassin is revealed, don't include it in the penalty
assassin_unrevealed = assassin not in revealed_words

# Encode words into vectors
red_vecs = model.encode(red_words) if red_words else np.array([])
blue_vecs = model.encode(blue_words) if blue_words else np.array([])
neutral_vecs = model.encode(neutral_words) if neutral_words else np.array([])
assassin_vec = model.encode([assassin])[0] if assassin_unrevealed else np.zeros(384)

# If no red words left, game is over
if len(red_words) == 0:
    hint = {"clue": "GAME_OVER", "number": 0}
    with open("hint.json", "w") as f:
        json.dump(hint, f)
    exit()

# Add more diverse candidate clues
# --------------------------------
# 3. Large candidate hint database
# --------------------------------
candidate_clues = [
    # Space / Science
    "space", "celestial", "astronomy", "planet", "moon", "star", "galaxy", "universe", "sky", "cosmos", "orbit",
    "asteroid", "comet", "meteor", "satellite", "blackhole", "nebula", "rocket", "telescope", "gravity", "wave",
    
    # Animals
    "animal", "pet", "mammal", "bird", "fish", "insect", "reptile", "amphibian", "wild", "domestic",
    "jungle", "farm", "oceanlife", "zoo", "predator", "prey", "feline", "canine", "aquatic", "flying",
    
    # Food
    "fruit", "vegetable", "meat", "dairy", "snack", "drink", "dessert", "sweet", "spicy", "cereal",
    "bread", "pizza", "pasta", "cake", "chocolate", "icecream", "soup", "salad", "coffee", "tea", "juice",
    
    # Objects
    "tool", "furniture", "vehicle", "transport", "clothing", "electronics", "instrument", "book", "paper", "toy",
    "weapon", "device", "appliance", "utensil", "machine", "phone", "computer", "keyboard", "pen", "pencil",
    
    # Nature
    "nature", "plant", "tree", "flower", "forest", "river", "mountain", "ocean", "beach", "desert",
    "weather", "sun", "rain", "storm", "snow", "wind", "cloud", "rock", "earth", "sky", "water",
    
    # Music / Art
    "music", "instrument", "painting", "drawing", "song", "dance", "theater", "poem", "composer", "band",
    "concert", "melody", "harmony", "lyric", "rhythm", "art", "sculpture", "canvas", "brush", "stage", "sound",
    
    # Misc / Concepts
    "color", "number", "shape", "emotion", "job", "profession", "school", "holiday", "festival", "party",
    "game", "sport", "competition", "travel", "city", "country", "history", "law", "politics", "technology",
    "internet", "social", "computer", "science", "medicine", "spaceflight", "myth", "legend", "story", "fiction",
    "time", "energy", "force", "light", "heat", "cold", "speed", "high", "low", "big", "small", "long", "short",
    "wide", "narrow", "thick", "thin", "hard", "soft", "sharp", "dull", "bright", "dark", "clear", "cloudy",
    "wet", "dry", "hot", "cool", "loud", "quiet", "fast", "heavy", "light", "strong", "weak", "smooth", "rough"
]

# -----------------------------
# 3a. Remove candidate clues that exactly match any board word
# -----------------------------
all_board_words = set(red_words + blue_words + neutral_words + ([assassin] if assassin_unrevealed else []))
candidate_clues = [c for c in candidate_clues if c.upper() not in all_board_words]

candidate_vecs = model.encode(candidate_clues)

# -----------------------------
# 4. Cosine similarity
# -----------------------------
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# -----------------------------
# 5. Improved scoring function for more literal connections
# -----------------------------
def score_clue(clue_vec):
    # Similarity to RED words (only unrevealed)
    similarities = [cosine_similarity(clue_vec, rv) for rv in red_vecs] if len(red_vecs) > 0 else []
    red_score = sum(similarities)
    
    # Reward clues that connect multiple RED words
    multi_word_bonus = sum(1 for sim in similarities if sim > 0.5) * 0.5  # +0.5 for each strong connection
    
    # Similarity to "bad" words (BLUE + NEUTRAL + ASSASSIN, unrevealed only)
    bad_vecs = list(blue_vecs) + list(neutral_vecs) + ([assassin_vec] if assassin_unrevealed else [])
    penalty = np.mean([cosine_similarity(clue_vec, bv) for bv in bad_vecs]) if bad_vecs else 0
    
    # Final score
    return red_score + multi_word_bonus - penalty

# -----------------------------
# 6. Pick the best clue
# -----------------------------
scores = [score_clue(cv) for cv in candidate_vecs]
best_index = int(np.argmax(scores))
best_clue_vec = candidate_vecs[best_index]
best_clue = str(candidate_clues[best_index]).upper()

# -----------------------------
# 7. Determine number of words it safely connects (dynamic threshold)
# -----------------------------
max_sim = max(cosine_similarity(best_clue_vec, rv) for rv in red_vecs) if len(red_vecs) > 0 else 0
threshold = 0.8 * max_sim
number = sum(1 for rv in red_vecs if cosine_similarity(best_clue_vec, rv) > threshold) if len(red_vecs) > 0 else 0
number = int(number)

# -----------------------------
# 8. Prefer clues that connect >1 word
# -----------------------------
if number == 1 and len(red_vecs) > 1:
    sorted_indices = np.argsort(scores)[::-1]  # descending order
    for idx in sorted_indices:
        vec = candidate_vecs[idx]
        max_sim_tmp = max(cosine_similarity(vec, rv) for rv in red_vecs)
        threshold_tmp = 0.8 * max_sim_tmp
        num_tmp = sum(1 for rv in red_vecs if cosine_similarity(vec, rv) > threshold_tmp)
        if num_tmp > 1:
            best_index = idx
            best_clue_vec = candidate_vecs[best_index]
            best_clue = str(candidate_clues[best_index]).upper()
            number = int(num_tmp)
            break

# Ensure number is at least 1 for unrevealed words
if number == 0 and len(red_vecs) > 0:
    number = 1

# Pick from top 3 best clues for variation
top_k = 3
top_indices = np.argsort(scores)[::-1][:top_k]
best_index = random.choice(top_indices)
best_clue = str(candidate_clues[best_index]).upper()

# Recalculate number for the selected clue
best_clue_vec = candidate_vecs[best_index]
max_sim = max(cosine_similarity(best_clue_vec, rv) for rv in red_vecs) if len(red_vecs) > 0 else 0
threshold = 0.8 * max_sim
number = sum(1 for rv in red_vecs if cosine_similarity(best_clue_vec, rv) > threshold) if len(red_vecs) > 0 else 0
if number == 0 and len(red_vecs) > 0:
    number = 1

# -----------------------------
# 9. Write hint.json safely
# -----------------------------
hint = {
    "clue": best_clue,
    "number": number
}

with open("hint.json", "w") as f:
    json.dump(hint, f)

# Print for debugging
print(f"AI Hint: {best_clue} ({number}) - Unrevealed RED words: {red_words}")
