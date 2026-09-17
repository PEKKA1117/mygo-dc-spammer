"""Turning a model Candidate into something Discord can display."""

from __future__ import annotations

import discord

from .engine import Candidate

EMBED_COLOR = discord.Color.from_rgb(126, 160, 219)


def build_reply(candidate: Candidate, style: str) -> dict:
    """Kwargs for `Message.reply(...)` / `send(...)` in the requested style.

    - image: bare image URL, so Discord unfurls it as a plain meme post
    - embed: image plus the quote as a caption
    - text:  quote only, no image
    """
    if style == "text":
        return {"content": candidate.quote}

    if style == "embed":
        embed = discord.Embed(description=candidate.quote, color=EMBED_COLOR)
        if candidate.image_url:
            embed.set_image(url=candidate.image_url)
        embed.set_footer(text=f"{candidate.probability:.1f}% · MyGOChat")
        return {"embed": embed}

    return {"content": candidate.image_url or candidate.quote}


def build_candidates_embed(candidates: list[Candidate], query: str) -> discord.Embed:
    """A ranked list of the model's top-k guesses, previewing the best one."""
    embed = discord.Embed(
        title="MyGOChat candidates",
        description=f"> {discord.utils.escape_markdown(query)[:200]}",
        color=EMBED_COLOR,
    )
    for index, candidate in enumerate(candidates, start=1):
        value = f"{candidate.probability:.2f}%"
        if candidate.image_url:
            value += f" · [image]({candidate.image_url})"
        embed.add_field(name=f"{index}. {candidate.quote}", value=value, inline=False)
    if candidates and candidates[0].image_url:
        embed.set_image(url=candidates[0].image_url)
    return embed


__all__ = ["EMBED_COLOR", "build_candidates_embed", "build_reply"]
