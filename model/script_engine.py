"""
model/script_engine.py
Generates and improves video scripts.

Pipeline:
  1. Generate raw script from multilingual templates
  2. Optionally improve via HuggingFace NLP (grammar + clarity)
  3. Split into timed segments
  4. Trim to max duration
"""

from __future__ import annotations

import re
from typing import Dict, List

from utils.config import get_config
from utils.helpers import estimate_duration, normalize_unicode
from utils.logger import get_logger

log = get_logger("model.script")

# ── Multilingual templates ─────────────────────────────────────────────────────

TEMPLATES: Dict[str, Dict[str, object]] = {
    "ar": {
        "news": lambda t: f"""مرحباً بكم في نشرتنا الإخبارية. اليوم نتناول موضوعاً يستحق الاهتمام، وهو {t}.

في ظل التطورات المتسارعة التي يشهدها عالمنا، يبرز {t} كواحدة من أبرز القضايا التي تؤثر على ملايين البشر.

الخبراء والمحللون يؤكدون أن {t} يمر بمرحلة تحول حقيقي. فالأرقام تتحدث بوضوح عن مدى التأثير والأهمية.

ما يميز هذه المرحلة هو الوعي المتزايد بهذا الملف. فالمجتمعات باتت تتفاعل مع {t} بصورة لم يسبق لها مثيل.

في الختام، يبقى {t} موضوعاً حيوياً يستحق متابعتكم. شاركونا آراءكم في التعليقات.""",

        "story": lambda t: f"""دعوني أحكي لكم قصة ملهمة عن {t}.

في البداية لم يكن أحد يعلم أن {t} سيصبح بهذه الأهمية. كان مجرد فكرة صغيرة في ذهن شخص يؤمن بالمستحيل.

ثم جاءت اللحظة الفارقة التي غيرت كل شيء. واجه الطريق تحديات كبيرة، لكن مع كل عقبة جاء إلهام جديد.

والآن ندرك لماذا يظل {t} قصة ملهمة تستحق أن تُروى.""",

        "facts": lambda t: f"""هل تعلم؟ اليوم سنكشف لك حقائق مذهلة عن {t}.

الحقيقة الأولى: {t} يرتبط بتاريخ الإنسانية ارتباطاً وثيقاً يمتد لآلاف السنين.

الحقيقة الثانية: دراسات علمية حديثة أثبتت أن {t} يؤثر على مليارات البشر بطرق غير متوقعة.

الحقيقة الثالثة: الأكثر إثارة أن {t} يخفي طبقات من التعقيد لا تزال تُكتشف حتى اليوم.

أيها الحقائق فاجأك أكثر؟ شاركنا في التعليقات.""",

        "education": lambda t: f"""أهلاً بكم. سنتعرف اليوم على {t} بطريقة بسيطة وممتعة.

ما هو {t} بالضبط؟ هو مفهوم يمتد تأثيره في حياتنا بطرق كثيرة لا نلاحظها أحياناً.

المبدأ الأساسي الذي يحكم {t} هو التوازن والتكامل بين عناصره، حيث يؤدي كل جزء دوره المحدد.

تذكر: {t} ليس مجرد نظرية، بل هو أداة عملية تساعدنا على فهم العالم من حولنا.""",
    },

    "en": {
        "news": lambda t: f"""Welcome to our news segment. Today we cover an important topic: {t}.

In a rapidly changing world, {t} has emerged as one of the most significant issues affecting millions globally.

Experts and analysts confirm that {t} is undergoing a real transformation. Data speaks clearly about its growing impact.

What makes this moment unique is the rising awareness around {t}. Communities worldwide are engaging like never before.

In conclusion, {t} remains a topic worth following. Share your thoughts in the comments.""",

        "story": lambda t: f"""Let me tell you an inspiring story about {t}.

At first, no one imagined that {t} would become this important. It started as a small idea in the mind of a true believer.

Then came the turning point that changed everything. Every obstacle brought new inspiration and resilience.

Now we understand why {t} remains a story worth telling, from generation to generation.""",

        "facts": lambda t: f"""Did you know? Today we reveal amazing facts about {t} that will surprise you.

Fact one: {t} is deeply connected to human history, stretching back thousands of years.

Fact two: Recent scientific studies show that {t} affects billions of people in unexpected ways.

Fact three: Most amazingly, {t} hides layers of complexity still being discovered today.

Which fact surprised you most? Tell us in the comments.""",

        "education": lambda t: f"""Welcome. Today we explore {t} in a simple and engaging way.

What exactly is {t}? It is a concept whose influence extends into our daily lives in ways we often overlook.

The core principle governing {t} is balance and integration between its elements, each playing a role.

Remember: {t} is not just theory — it is a practical tool that helps us understand the world around us.""",
    },

    "fr": {
        "news":      lambda t: f"Bienvenue dans notre journal. Aujourd'hui nous parlons de {t}.\n\n{t} est devenu un sujet important affectant des millions de personnes.\n\nLes experts confirment que {t} traverse une transformation réelle.\n\nEn conclusion, {t} reste crucial. Partagez vos avis.",
        "story":     lambda t: f"Laissez-moi vous raconter une histoire sur {t}.\n\nAu début personne n'imaginait que {t} deviendrait si important.\n\nPuis vint le moment décisif. Chaque défi apportait une nouvelle inspiration.\n\nAujourd'hui {t} reste une histoire qui mérite d'être racontée.",
        "facts":     lambda t: f"Le saviez-vous? Voici des faits étonnants sur {t}.\n\nFait 1: {t} est lié à l'histoire humaine depuis des millénaires.\n\nFait 2: Des études récentes montrent que {t} affecte des milliards de personnes.\n\nFait 3: {t} cache des couches de complexité encore découvertes aujourd'hui.",
        "education": lambda t: f"Bienvenue. Explorons {t} simplement.\n\nQu'est-ce que {t}? Un concept dont l'influence s'étend dans notre vie quotidienne.\n\nLe principe de {t} est l'équilibre entre ses éléments.\n\n{t} est un outil pratique pour comprendre le monde.",
    },

    "es": {
        "news":      lambda t: f"Bienvenidos al noticiero. Hoy hablamos de {t}.\n\n{t} se ha convertido en un tema importante a nivel mundial.\n\nLos expertos confirman que {t} atraviesa una transformación real.\n\nEn conclusión, {t} merece su atención. Comparta sus opiniones.",
        "story":     lambda t: f"Déjame contarte una historia sobre {t}.\n\nAl principio nadie imaginaba que {t} sería tan importante.\n\nLuego llegó el momento decisivo. Cada desafío trajo nueva inspiración.\n\nHoy {t} sigue siendo una historia que vale la pena contar.",
        "facts":     lambda t: f"¿Sabías esto? Hoy revelamos hechos sobre {t}.\n\nHecho 1: {t} está conectado con la historia humana desde hace milenios.\n\nHecho 2: Estudios muestran que {t} afecta a miles de millones.\n\nHecho 3: {t} esconde capas de complejidad aún por descubrir.",
        "education": lambda t: f"Bienvenidos. Hoy exploramos {t} de manera sencilla.\n\n¿Qué es {t}? Un concepto presente en nuestra vida diaria.\n\nEl principio de {t} es el equilibrio entre sus partes.\n\n{t} es una herramienta práctica para entender el mundo.",
    },

    "de": {
        "news":      lambda t: f"Willkommen zur Nachrichtensendung. Heute sprechen wir über {t}.\n\n{t} hat sich zu einem wichtigen Thema entwickelt, das Millionen betrifft.\n\nExperten bestätigen, dass {t} einen Wandel durchläuft.\n\nAbschließend bleibt {t} ein wichtiges Thema. Teilen Sie Ihre Meinung.",
        "story":     lambda t: f"Lassen Sie mich eine Geschichte über {t} erzählen.\n\nAm Anfang ahnte niemand, dass {t} so wichtig werden würde.\n\nDann kam der entscheidende Wendepunkt. Jede Herausforderung brachte neue Inspiration.\n\nHeute bleibt {t} eine Geschichte, die es wert ist erzählt zu werden.",
        "facts":     lambda t: f"Wussten Sie das? Heute enthüllen wir Fakten über {t}.\n\nFakt 1: {t} ist seit Jahrtausenden mit der Menschheitsgeschichte verbunden.\n\nFakt 2: Studien zeigen, dass {t} Milliarden von Menschen beeinflusst.\n\nFakt 3: {t} verbirgt Komplexitätsebenen, die heute noch entdeckt werden.",
        "education": lambda t: f"Willkommen. Heute erkunden wir {t} auf einfache Weise.\n\nWas ist {t}? Ein Konzept, das unseren Alltag auf viele Arten beeinflusst.\n\nDas Grundprinzip von {t} ist das Gleichgewicht zwischen seinen Elementen.\n\n{t} ist ein praktisches Werkzeug, um die Welt zu verstehen.",
    },
}


class ScriptEngine:
    """
    Generates, improves, and segments video scripts.
    """

    def __init__(self) -> None:
        self.cfg    = get_config()
        self._hf    = None

    # ── Public ─────────────────────────────────────────────────────────────────

    def generate(self, topic: str, video_type: str = "news",
                 language: str = "ar") -> dict:
        """
        Generate a complete script dict:
        {
            full_text, segments, total_duration,
            topic, video_type, language, title
        }
        """
        log.info("Generating %s script in %s: %s", video_type, language, topic)

        # 1. Raw template
        raw = self._from_template(topic, video_type, language)

        # 2. Optionally improve via HF
        if self.cfg.hf.use_hf_text:
            improved = self._hf_improve(raw, language)
            if improved and len(improved) > 100:
                raw = improved
                log.info("Script improved via HF")

        # 3. Segment
        segments = self._segment(raw)
        segments = self._trim(segments, self.cfg.video.max_duration_sec)
        total    = sum(s["duration"] for s in segments)

        log.info("Script: %d segments, ~%.0fs", len(segments), total)
        return {
            "full_text":      "\n\n".join(s["text"] for s in segments),
            "segments":       segments,
            "total_duration": total,
            "topic":          topic,
            "video_type":     video_type,
            "language":       language,
            "title":          f"{video_type.upper()}: {topic}",
        }

    # ── Private ────────────────────────────────────────────────────────────────

    def _from_template(self, topic: str, video_type: str, language: str) -> str:
        lang_t = TEMPLATES.get(language, TEMPLATES["ar"])
        fn     = lang_t.get(video_type, lang_t.get("news"))
        return fn(topic).strip()  # type: ignore[operator]

    def _hf_improve(self, script: str, language: str) -> str:
        if self._hf is None:
            from hf.hf_model_loader import get_loader
            self._hf = get_loader()
        return self._hf.improve_script(script, language)

    def _segment(self, text: str) -> List[dict]:
        """Split script into paragraph-level segments with duration estimates."""
        paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
        segs  = []
        for p in paras:
            lines = [l.strip() for l in p.splitlines()
                     if l.strip() and not (l.strip().endswith(":") and len(l.strip()) < 30)]
            if not lines:
                continue
            body = " ".join(lines)
            body = normalize_unicode(body)
            dur  = max(self.cfg.video.min_segment_sec, estimate_duration(body))
            segs.append({"text": body, "duration": dur})
        return segs

    def _trim(self, segments: List[dict], max_dur: float) -> List[dict]:
        """Remove middle segments if total exceeds max_dur."""
        total = sum(s["duration"] for s in segments)
        while total > max_dur and len(segments) > 2:
            removed = segments.pop(-2)
            total  -= removed["duration"]
        return segments
