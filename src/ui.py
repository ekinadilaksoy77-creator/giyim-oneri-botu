"""Renkli terminal arayüzü (rich). rich kurulu değilse bot.py düz metin arayüzüne döner."""
from __future__ import annotations

import re

from rich import box
from rich.console import Console, Group
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

import config
from src.bot import HELP, Response, is_exit

console = Console()


def _banner(bot) -> Panel:
    mode = f"LLM: {bot.llm.model}" if bot.llm.available else "LLM yok, kural tabanlı mod"
    body = Text.assemble(
        ("Giyim Öneri Botu\n", "bold cyan"),
        (f"[{mode}]\n\n", "dim"),
        ("Örnekler\n", "bold"),
        ("  • lacivert gömlek, M beden, bütçem 1500\n", "green"),
        ("  • müşteri <id>      ", "green"), ("kayıtlı müşteri önerisi\n", "dim"),
        ("  • paket kazak       ", "green"), ("birlikte alınan ürünler\n", "dim"),
        ("  • çık               ", "green"), ("(veya q, Ctrl+C) çıkış", "dim"),
    )
    return Panel(body, box=box.ROUNDED, border_style="cyan", padding=(1, 2))


def _short(reason: str) -> str:
    """Uzun grup adını her satırda tekrarlamamak için gerekçeyi kısaltır (grup adı üstte zaten yazıyor)."""
    return re.sub(r"Benzer tercihleri olan müşteri grubunda \('.*?'\) bu ürünü alma oranı:", "Bu gruptaki alma oranı:", reason)


def _table(resp: Response) -> Table:
    t = Table(box=box.SIMPLE_HEAD, header_style="bold magenta", expand=True, show_lines=False)
    t.add_column("#", justify="right", width=3)
    t.add_column("Ürün", style="bold")
    t.add_column("Fiyat", justify="right")
    t.add_column("Neden önerildi")
    for i, r in enumerate(resp.result.recs, 1):
        name = f"{config.CATEGORY_TR.get(r.category, r.category).capitalize()} - {r.colour}"
        why = "\n".join(f"• {_short(x)}" for x in r.reasons)
        t.add_row(str(i), name, f"~{r.price:.0f}", why)
    return t


def render(resp: Response, show_explanation: bool = True) -> None:
    if resp.result is None:
        console.print(Panel(resp.text, border_style="yellow", box=box.ROUNDED))
        return
    res = resp.result
    if not res.recs:
        console.print(Panel("Uygun ürün bulunamadı.", border_style="red"))
        return
    parts = []
    if res.segment_name:
        parts.append(Text(f"Tahmini müşteri grubu: {res.segment_name}", style="cyan"))
    parts.append(_table(resp))
    for n in res.notes:
        parts.append(Text(f"⚠ {n}", style="yellow"))
    if resp.explanation and show_explanation:
        parts.append(Panel(resp.explanation, title="Danışman notu", border_style="green", box=box.ROUNDED))
    console.print(Panel(Group(*parts), title=resp.title or "Öneriler", border_style="cyan", box=box.ROUNDED))


def run(bot) -> None:
    console.print(_banner(bot))
    while True:
        try:
            msg = console.input("\n[bold blue]Siz ›[/] ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not msg:
            continue
        if is_exit(msg):
            break
        with console.status("[cyan]Düşünüyorum...", spinner="dots"):
            resp = bot.respond(msg)
        # Şablon (LLM'siz) açıklama notları tekrar eder; yalnızca gerçek LLM açıklaması ayrı kutuda gösterilir
        render(resp, show_explanation=bot.llm.available)
        if bot.llm.last_error:
            console.print(f"[dim](LLM uyarısı, yedek mod kullanıldı: {bot.llm.last_error})[/]")
    console.print("\n[bold cyan]Görüşmek üzere![/]")
