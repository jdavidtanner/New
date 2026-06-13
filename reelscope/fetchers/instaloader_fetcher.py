"""Fallback fetcher using the instaloader library.

Free and pure-Python, but Instagram frequently changes its internals so this
can break for short stretches. Good as a second option behind yt-dlp.

Requires ``pip install instaloader``.
"""

from __future__ import annotations

from datetime import datetime, timezone

from ..models import ChannelData, Reel
from .base import Fetcher, FetcherError, normalize_handle


class InstaloaderFetcher(Fetcher):
    name = "instaloader"

    def __init__(self, login_user: str | None = None, **_: object) -> None:
        # login_user: optionally reuse a saved instaloader session for this user.
        self.login_user = login_user

    def fetch(self, handle_or_url: str, limit: int = 30) -> ChannelData:
        try:
            import instaloader  # noqa: PLC0415 (optional dependency)
        except ImportError as e:
            raise FetcherError(
                "instaloader is not installed. Install it with:  pip install instaloader"
            ) from e

        handle = normalize_handle(handle_or_url)
        L = instaloader.Instaloader(
            download_pictures=False,
            download_videos=False,
            download_comments=False,
            save_metadata=False,
            quiet=True,
        )
        if self.login_user:
            try:
                L.load_session_from_file(self.login_user)
            except FileNotFoundError as e:
                raise FetcherError(
                    f"No saved instaloader session for '{self.login_user}'. "
                    f"Create one with:  instaloader --login {self.login_user}"
                ) from e

        try:
            profile = instaloader.Profile.from_username(L.context, handle)
        except Exception as e:  # instaloader raises several specific types
            raise FetcherError(f"Could not load profile '{handle}': {e}") from e

        reels: list[Reel] = []
        for post in profile.get_posts():
            if not post.is_video:
                continue
            reels.append(
                Reel(
                    url=f"https://www.instagram.com/reel/{post.shortcode}/",
                    caption=post.caption or "",
                    view_count=getattr(post, "video_view_count", None),
                    like_count=post.likes,
                    comment_count=post.comments,
                    duration_sec=getattr(post, "video_duration", None),
                    timestamp=int(post.date_utc.replace(tzinfo=timezone.utc).timestamp()),
                    thumbnail=post.url,
                )
            )
            if len(reels) >= limit:
                break

        return ChannelData(
            handle=handle,
            source=self.name,
            fetched_at=datetime.now(timezone.utc).isoformat(),
            reels=reels,
            follower_count=profile.followers,
            bio=profile.biography or "",
        )
