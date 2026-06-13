"""ReelScope — research an Instagram channel's reels, learn what makes them
work, and turn that into ready-to-use Higgsfield briefs for your own business.

The pipeline is three stages:

1. fetch    — pull a channel's reels (captions, views, likes, dates, duration)
              from a pluggable source (yt-dlp, instaloader, manual, apify).
2. analyze  — rank reels by performance and extract the repeatable patterns
              (hooks, formats, cadence, length, hashtags, CTAs).
3. brief    — convert the winning patterns into Higgsfield prompts + shot lists,
              reskinned to *your* business so you duplicate the structure,
              not the content.
"""

from .models import Reel, ChannelData, ChannelAnalysis

__all__ = ["Reel", "ChannelData", "ChannelAnalysis"]
__version__ = "0.1.0"
