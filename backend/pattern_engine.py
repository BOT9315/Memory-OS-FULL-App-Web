"""
Phase 3: Pattern Engine
The unique feature — proactive life intelligence.
Detects patterns humans can't see in their own lives.
"""
import json
from datetime import datetime, timedelta
from typing import List, Dict, Optional


class PatternEngine:
    def __init__(self, claude_client):
        self.client = claude_client

    def get_proactive_message(self, patterns: Dict, entries: List[Dict]) -> Optional[str]:
        """Generate a proactive insight to open a conversation with"""
        if not patterns or not entries:
            return None

        insights = []

        # Check for patterns
        if patterns.get("patterns"):
            insights.append(f"I noticed: {patterns['patterns'][0]}")

        # Check relationship gaps
        for rel in (patterns.get("relationships") or []):
            if rel.get("last_mentioned"):
                try:
                    last = datetime.fromisoformat(rel["last_mentioned"])
                    days = (datetime.now() - last).days
                    if days > 21:
                        insights.append(f"You haven't mentioned {rel['name']} in {days} days.")
                except:
                    pass

        # Check for stalled goals
        for g in (patterns.get("goals_detected") or []):
            if g.get("status") == "stalled":
                insights.append(f"Your goal '{g['goal']}' hasn't come up lately.")

        # Check concern areas
        if patterns.get("concern_areas"):
            insights.append(f"'{patterns['concern_areas'][0]}' keeps appearing in your entries.")

        return insights[0] if insights else None

    def score_entry_mood(self, text: str) -> int:
        """Quick heuristic mood score 1-10 from text"""
        positive = ["happy", "excited", "great", "amazing", "proud", "achieved", "love",
                    "wonderful", "fantastic", "joy", "grateful", "motivated", "energized",
                    "accomplished", "confident", "inspired", "hopeful", "peaceful"]
        negative = ["sad", "anxious", "stressed", "worried", "angry", "frustrated", "depressed",
                    "exhausted", "overwhelmed", "lonely", "scared", "failed", "hopeless",
                    "terrible", "awful", "hate", "stuck", "lost", "empty"]

        text_lower = text.lower()
        pos_count = sum(1 for w in positive if w in text_lower)
        neg_count = sum(1 for w in negative if w in text_lower)

        base = 5
        score = base + (pos_count * 0.8) - (neg_count * 0.9)
        return max(1, min(10, round(score)))

    def extract_people_mentioned(self, text: str) -> List[str]:
        """Simple name extraction — proper names (capitalized, not sentence start)"""
        import re
        words = text.split()
        names = []
        for i, word in enumerate(words):
            clean = re.sub(r'[^a-zA-Z]', '', word)
            if (len(clean) > 2 and clean[0].isupper() and clean[1:].islower()
                    and i > 0  # not sentence start
                    and clean not in {"The", "A", "An", "And", "But", "So", "My",
                                      "I", "We", "They", "He", "She", "It", "This",
                                      "That", "Today", "Yesterday", "Monday", "Tuesday",
                                      "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"}):
                names.append(clean)
        return list(set(names))

    def detect_weekly_pattern(self, entries: List[Dict]) -> Optional[str]:
        """Detect if mood is consistently worse/better on certain days"""
        from collections import defaultdict
        day_scores = defaultdict(list)

        for entry in entries:
            try:
                dt = datetime.fromisoformat(entry["timestamp"])
                day_name = dt.strftime("%A")
                score = self.score_entry_mood(entry["user_message"])
                day_scores[day_name].append(score)
            except:
                pass

        if not day_scores:
            return None

        day_avgs = {day: sum(scores) / len(scores)
                    for day, scores in day_scores.items() if len(scores) >= 2}

        if not day_avgs:
            return None

        best_day = max(day_avgs, key=day_avgs.get)
        worst_day = min(day_avgs, key=day_avgs.get)

        if day_avgs[best_day] - day_avgs[worst_day] > 2:
            return f"You seem most positive on {best_day}s and most drained on {worst_day}s"

        return None
