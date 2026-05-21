#!/usr/bin/env python3
from gz_topic_tools import list_gz_topics

print("Listing Gazebo topics using 'gz topic -l' / 'ign topic -l'...")
topics = list_gz_topics()
if not topics:
    print("❌ No topics found. Start PX4/Gazebo first, then run this again.")
else:
    print(f"✅ Found {len(topics)} topics:\n")
    for t in topics:
        low = t.lower()
        tag = ""
        if "depth" in low and "image" in low:
            tag = "   <-- possible DEPTH_TOPIC"
        elif "camera" in low and "image" in low:
            tag = "   <-- possible RGB_TOPIC"
        print(t + tag)
