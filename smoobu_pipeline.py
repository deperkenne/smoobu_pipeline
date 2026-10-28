"""
Pipeline Smoobu multi-agents (LangGraph + LangSmith).

Variables d'environnement (à définir dans PyCharm : Run > Edit Configurations > Environment variables)
  ANTHROPIC_API_KEY      clé API Anthropic (obligatoire)
  GMAIL_APP_PASSWORD     mot de passe d'application Google (obligatoire)
  GMAIL_USER             adresse Gmail (défaut : tchofodep@gmail.com)
  SPRING_BOOT_URL        URL du backend (défaut : http://localhost:8080)
  LANGSMITH_API_KEY      clé LangSmith (ou LANGCHAIN_API_KEY, ancien nom accepté)
  LANGSMITH_PROJECT      nom du projet LangSmith (défaut : smoobu-pipeline)
  LANGSMITH_ENDPOINT     UNIQUEMENT si votre compte est en Europe : https://eu.api.smith.langchain.com
  ANTHROPIC_MODEL        modèle (défaut : claude-sonnet-5-5)
  ANTHROPIC_MAX_TOKENS   limite de sortie du modèle (défaut : 8000)
  USE_SAMPLE_EMAIL       "true" pour tester avec un faux e-mail quand la boîte est vide

Commandes
  python smoobu_pipeline.py             lance un cycle du pipeline
  python smoobu_pipeline.py --graph     affiche le graphe (Mermaid) sans rien exécuter
  python smoobu_pipeline.py --unblock   lève le blocage après une correction humaine
"""
import argparse
import email
import hashlib
import html
import imaplib
import json
import os
import re
import smtplib
import subprocess
import sys
import time
from email.message import EmailMessage
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, TypedDict


# --------------------------------------------------------------------------- #
# 1. LANGSMITH : doit être configuré AVANT les imports langchain / langgraph  #
# --------------------------------------------------------------------------- #
def setup_langsmith() -> bool:
	"""Normalise les variables (anciens noms LANGCHAIN_* et nouveaux LANGSMITH_*)."""
	key = os.getenv("LANGSMITH_API_KEY") or os.getenv("LANGCHAIN_API_KEY")
	if not key:
		return False
	project = os.getenv("LANGSMITH_PROJECT") or os.getenv("LANGCHAIN_PROJECT") or "smoobu-pipeline"
	endpoint = os.getenv("LANGSMITH_ENDPOINT") or os.getenv("LANGCHAIN_ENDPOINT")

	for prefix in ("LANGSMITH", "LANGCHAIN"):
		os.environ[f"{prefix}_API_KEY"] = key
		os.environ[f"{prefix}_PROJECT"] = project
		if endpoint:
			os.environ[f"{prefix}_ENDPOINT"] = endpoint
	os.environ["LANGSMITH_TRACING"] = "true"
	os.environ["LANGCHAIN_TRACING_V2"] = "true"
	return True


LANGSMITH_ON = setup_langsmith()

import httpx  # noqa: E402
from langchain_anthropic import ChatAnthropic  # noqa: E402
from langgraph.graph import END, StateGraph  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402

# --------------------------------------------------------------------------- #
# 2. CONFIGURATION                                                            #
# --------------------------------------------------------------------------- #
MODEL = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-5-5")
MAX_TOKENS = int(os.getenv("ANTHROPIC_MAX_TOKENS", "8000"))
GMAIL_USER = os.getenv("GMAIL_USER", "tchofodep@gmail.com")
GMAIL_APP_PASSWORD = os.getenv("GMAIL_APP_PASSWORD")
SPRING_BOOT_URL = os.getenv("SPRING_BOOT_URL", "http://localhost:8080")
ALERT_TO = os.getenv("ALERT_TO", "tchofodep@gmail.com")
USE_SAMPLE_EMAIL = os.getenv("USE_SAMPLE_EMAIL", "").lower() == "true"
SMOOBU_SENDER = os.getenv("SMOOBU_SENDER", "service@smoobu.com")  # seul expéditeur accepté

BLOCK_FILE = Path("pipeline_blocked.json")  # présent = pipeline bloqué
REVIEW_CACHE = Path(".last_review.json")  # dernier diff analysé + verdict
MAX_RETRIES = 2  # relances automatiques max par agent
RETRYABLE_NODES = {"pipeline_email", "load_to_db"}  # seuls agents relançables automatiquement

SAMPLE_EMAIL = """
Neue Buchung: Loft am Pegnitztal Nürnberg: 06.11.26 – 08.11.26 Šárka Pekárková
Du hast eine neue Buchung!
Loft am Pegnitztal Nürnberg
Anreise: Fr., 06.11.26
Abreise: So., 08.11.26
Gast: Šárka Pekárková
Gastnachricht: Buchungsnummer: 6041336634
Gastnachricht: ** THIS RESERVATION HAS BEEN PRE-PAID ** BOOKING NOTE : Payment charge is EUR 2.87336
Weitere Angaben: booker_is_genius
Anzahlung: 205.24EUR
"""


# --------------------------------------------------------------------------- #
# 3. SCHÉMAS                                                                  #
# --------------------------------------------------------------------------- #
class ReviewVerdict(BaseModel):
	is_clean: bool = Field(description="True si le code est propre, False s'il contient des erreurs bloquantes")
	reason: str = Field(description="Explication technique de l'évaluation")


class SmoobuBooking(BaseModel):
	property_name: str
	check_in: str
	check_out: str
	guest_name: str
	booking_number: str
	prepaid: bool
	booking_fee: Optional[float] = None
	deposit_amount: float
	currency: str = "EUR"
	notes: Optional[str] = None


class DebugAnalysis(BaseModel):
	root_cause: str = Field(description="Cause racine de l'échec, en une ou deux phrases")
	can_auto_fix: bool = Field(
		description="True seulement si c'est une panne passagère qu'une simple relance peut résoudre "
					"(réseau, timeout, erreur serveur 5xx temporaire). False sinon.")
	explanation: str = Field(description="Analyse technique détaillée")
	solutions: List[str] = Field(description="Étapes concrètes pour corriger, si intervention humaine nécessaire")


class AgentState(TypedDict, total=False):
	code_diff: str
	test_output: str
	email_ids: List[str]
	raw_emails: List[str]
	transformed_data: List[Dict[str, Any]]
	batched_payloads: List[List[Dict[str, Any]]]
	current_error: Optional[str]
	error_kind: str  # "transient" (relançable) ou "permanent"
	failed_node: Optional[str]
	attempts: Dict[str, int]  # relances par agent
	requires_human: bool
	debug_report: Dict[str, Any]
	notification_sent: bool
	no_new_commit: bool


# --------------------------------------------------------------------------- #
# 4. OUTILS D'ENTRÉE/SORTIE (séparés des agents pour pouvoir les tester)      #
# --------------------------------------------------------------------------- #
class _Structured:
	"""Sortie structurée robuste.

	claude-sonnet-5-5 ne supporte pas l'appel d'outil forcé (méthode par défaut) : on utilise donc
	'json_schema'. max_tokens est large car le raisonnement du modèle consomme aussi des jetons ;
	avec une limite trop basse, la réponse est coupée et des champs manquent (ex. 'solutions').
	"""

	def __init__(self, schema):
		self.schema = schema

	def invoke(self, prompt):
		llm = ChatAnthropic(model=MODEL, api_key=os.environ["ANTHROPIC_API_KEY"], max_tokens=MAX_TOKENS)
		try:
			return llm.with_structured_output(self.schema, method="json_schema").invoke(prompt)
		except Exception as first:
			print(f"Sortie structurée 'json_schema' impossible ({first!r:.200}). Essai par appel d'outil.")
			return llm.with_structured_output(self.schema).invoke(prompt)


def structured_llm(schema):
	return _Structured(schema)


def run_git_diff() -> str:
	r = subprocess.run(["git", "diff", "HEAD"], capture_output=True, text=True, check=True)
	return r.stdout.strip()


def run_pytest() -> Tuple[int, str]:
	r = subprocess.run([sys.executable, "-m", "pytest", "--maxfail=1", "--disable-warnings", "-q"],
					   capture_output=True, text=True)
	return r.returncode, r.stdout + "\n" + r.stderr


def _html_to_text(raw_html: str) -> str:
	"""Convertit le HTML en texte, une information par ligne (les e-mails Smoobu n'ont qu'une version HTML)."""
	t = re.sub(r"(?is)<(style|script|head).*?</\1>", " ", raw_html)
	t = re.sub(r"(?i)<br\s*/?>|</(p|div|tr|li|td|h\d)>", "\n", t)
	t = html.unescape(re.sub(r"<[^>]+>", " ", t))
	lines = (re.sub(r"[ \t\xa0]+", " ", line).strip() for line in t.splitlines())
	return "\n".join(line for line in lines if line)


def _plain_text(msg) -> str:
	"""Renvoie la version texte de l'e-mail, ou à défaut sa version HTML convertie en texte."""
	html_body = ""
	for p in msg.walk():
		if p.get_content_disposition() == "attachment":
			continue
		payload = p.get_payload(decode=True)
		if not payload:
			continue
		text = payload.decode(p.get_content_charset() or "utf-8", errors="ignore")
		if p.get_content_type() == "text/plain" and text.strip():
			return text
		if p.get_content_type() == "text/html" and not html_body:
			html_body = text
	return _html_to_text(html_body) if html_body else ""


def _imap_connect():
	m = imaplib.IMAP4_SSL("imap.gmail.com")
	m.login(GMAIL_USER, GMAIL_APP_PASSWORD)
	m.select("inbox")
	return m


def imap_fetch_unseen() -> Tuple[List[str], List[str]]:
	"""Lit les mails non lus SANS les marquer comme lus (BODY.PEEK). Retourne (uids, textes)."""
	mail = _imap_connect()
	uids, bodies = [], []
	try:
		# Seuls les e-mails envoyés par Smoobu : un e-mail de test ne bloque plus le pipeline.
		status, data = mail.uid("search", None, f'(UNSEEN FROM "{SMOOBU_SENDER}" SUBJECT "Neue Buchung")')
		if status == "OK" and data[0]:
			for uid in data[0].split():
				_, msg_data = mail.uid("fetch", uid, "(BODY.PEEK[])")
				for part in msg_data:
					if isinstance(part, tuple):
						text = _plain_text(email.message_from_bytes(part[1]))
						if text:
							uids.append(uid.decode())
							bodies.append(text)
	finally:
		try:
			mail.logout()
		except Exception:
			pass
	return uids, bodies


def imap_mark_seen(uids: List[str]) -> None:
	if not uids:
		return
	mail = _imap_connect()
	try:
		for uid in uids:
			mail.uid("store", uid, "+FLAGS", "\\Seen")
	finally:
		try:
			mail.logout()
		except Exception:
			pass


def post_batch(batch: List[Dict[str, Any]]) -> None:
	with httpx.Client(base_url=SPRING_BOOT_URL, timeout=15.0) as client:
		client.post("/api/reservations/batch", json=batch).raise_for_status()


def send_mail(subject: str, body: str) -> None:
	msg = EmailMessage()
	msg.set_content(body)
	msg["Subject"], msg["From"], msg["To"] = subject, GMAIL_USER, ALERT_TO
	with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
		server.login(GMAIL_USER, GMAIL_APP_PASSWORD)
		server.send_message(msg)


def fail(node: str, message: str, kind: str = "permanent") -> dict:
	return {"current_error": message, "failed_node": node, "error_kind": kind}


def ok(**extra) -> dict:
	return {"current_error": None, "failed_node": None, **extra}


# --------------------------------------------------------------------------- #
# 5. LES 8 AGENTS                                                             #
# --------------------------------------------------------------------------- #
def _read_review_cache() -> dict:
	try:
		return json.loads(REVIEW_CACHE.read_text(encoding="utf-8"))
	except (OSError, ValueError):
		return {}


def code_reviewer_node(state: AgentState) -> dict:
	print("\n--- [1] code_reviewer : analyse du git diff ---")
	try:
		diff = run_git_diff()
		if not diff:
			# Pas de modification : on n'appelle pas le LLM (économie de tokens).
			print("Aucun changement local : LLM non appelé, code validé par défaut.")
			return ok(code_diff="", no_new_commit=True)

		# Même diff que lors de la dernière revue : on réutilise le verdict sans rappeler le LLM.
		diff_hash = hashlib.sha256(diff.encode("utf-8")).hexdigest()
		cached = _read_review_cache()
		if cached.get("hash") == diff_hash:
			print("Diff inchangé depuis la dernière revue : LLM non appelé, verdict réutilisé.")
			if cached.get("is_clean"):
				return ok(code_diff=diff, no_new_commit=False)
			return {**fail("code_reviewer", f"Revue de code refusée : {cached.get('reason', '')}"),
					"code_diff": diff}

		prompt = f"""
			Tu es un Lead Software Engineer expert en architecture logicielle et revue de code.
			Analyse le git diff suivant :

			```diff
			{diff}
			```

			Vérifie la robustesse globale, l'architecture, le typage, la gestion des erreurs et l'absence de failles ou de bugs potentiels.
			Détermine si le code est propre (`is_clean = True`) ou s'il présente des anomalies bloquantes (`is_clean = False`).
			"""
		verdict = structured_llm(ReviewVerdict).invoke(prompt)
		REVIEW_CACHE.write_text(json.dumps({"hash": diff_hash, "is_clean": verdict.is_clean,
											"reason": verdict.reason}), encoding="utf-8")
		if verdict.is_clean:
			print(f"Code validé : {verdict.reason}")
			return ok(code_diff=diff, no_new_commit=False)
		return {**fail("code_reviewer", f"Revue de code refusée : {verdict.reason}"), "code_diff": diff}

	except Exception as e:
		return fail("code_reviewer", f"Erreur d'exécution du code_reviewer : {e!r}")


def run_test_node(state: AgentState) -> dict:
	print("--- [2] run_test : pytest ---")
	try:
		code, output = run_pytest()
		if code == 0:
			print("Tests unitaires OK.")
			return ok(test_output=output)
		reason = "aucun test trouvé" if code == 5 else f"code retour {code}"
		return {**fail("run_test", f"pytest a échoué ({reason}).\n{output[-3000:]}"), "test_output": output}
	except Exception as e:
		return fail("run_test", f"Erreur d'exécution de pytest : {e!r}")


def pipeline_email_node(state: AgentState) -> dict:
	print("--- [3] pipeline_email : lecture IMAP ---")
	try:
		uids, bodies = imap_fetch_unseen()
		if not bodies and USE_SAMPLE_EMAIL:
			print("Boîte vide : e-mail d'exemple injecté (USE_SAMPLE_EMAIL=true).")
			bodies = [SAMPLE_EMAIL]
		print(f"{len(bodies)} e-mail(s) Smoobu à traiter.")
		return ok(email_ids=uids, raw_emails=bodies)
	except imaplib.IMAP4.error as e:  # identifiants refusés, etc.
		return fail("pipeline_email", f"Erreur IMAP (identifiants ou configuration) : {e}", "permanent")
	except (OSError, imaplib.IMAP4.abort) as e:  # réseau, connexion coupée
		return fail("pipeline_email", f"Connexion IMAP impossible : {e!r}", "transient")
	except Exception as e:
		return fail("pipeline_email", f"Erreur inattendue (IMAP) : {e!r}")


# champ -> expression régulière (le premier groupe capturé est la valeur)
PATTERNS = {
	"Anreise": r"Anreise:\s*\w+\.,\s*(\d{2}\.\d{2}\.\d{2})",
	"Abreise": r"Abreise:\s*\w+\.,\s*(\d{2}\.\d{2}\.\d{2})",
	"Gast": r"Gast:\s*(.*)",
	"Buchungsnummer": r"Buchungsnummer:\s*(\d+)",
}


def data_transformer_node(state: AgentState) -> dict:
	print("--- [4] data_transformer : extraction des champs ---")
	results, problems = [], []
	for i, raw in enumerate(state.get("raw_emails", []), start=1):
		try:
			found = {}
			for field, pattern in PATTERNS.items():
				m = re.search(pattern, raw)
				if not m:
					raise ValueError(f"champ '{field}' introuvable")
				found[field] = m.group(1).strip()

			prop = re.search(r"Neue Buchung:\s*(.*?):", raw)
			fee = re.search(r"Payment charge is EUR\s*([\d\.]+)", raw)
			deposit = re.search(r"Anzahlung:\s*([\d\.]+)EUR", raw)
			notes = re.search(r"Weitere Angaben:\s*(.*)", raw)

			results.append(SmoobuBooking(
				property_name=prop.group(1).strip() if prop else "Loft am Pegnitztal Nürnberg",
				check_in=found["Anreise"], check_out=found["Abreise"],
				guest_name=found["Gast"], booking_number=found["Buchungsnummer"],
				prepaid="PRE-PAID" in raw,
				booking_fee=float(fee.group(1)) if fee else 0.0,
				deposit_amount=float(deposit.group(1)) if deposit else 0.0,
				notes=notes.group(1).strip() if notes else None,
			).model_dump())
		except Exception as e:
			# on ne met PAS le contenu du mail dans l'erreur (données personnelles)
			problems.append(f"e-mail n°{i} : {e}")

	if problems:
		return fail("data_transformer",
					"Extraction impossible pour : " + " ; ".join(problems) +
					". Le modèle d'e-mail Smoobu a peut-être changé : adaptez PATTERNS.")
	return ok(transformed_data=results)


def load_list_node(state: AgentState) -> dict:
	print("--- [5] load_list : dédoublonnage et lots ---")
	unique = list({b["booking_number"]: b for b in state.get("transformed_data", [])}.values())
	return ok(batched_payloads=[unique[i:i + 50] for i in range(0, len(unique), 50)])


def load_to_db_node(state: AgentState) -> dict:
	print("--- [6] load_to_db : envoi à l'API Spring Boot ---")
	try:
		for batch in state.get("batched_payloads", []):
			post_batch(batch)
		imap_mark_seen(state.get("email_ids", []))  # marqués lus SEULEMENT après succès
		print("Réservations insérées, e-mails marqués comme traités.")
		return ok()
	except httpx.HTTPStatusError as e:
		code = e.response.status_code
		kind = "transient" if code >= 500 or code in (408, 429) else "permanent"
		return fail("load_to_db", f"HTTP {code} sur {e.request.url} : {e.response.text[:500]}", kind)
	except httpx.TransportError as e:  # connexion refusée, timeout...
		return fail("load_to_db", f"Backend injoignable ({SPRING_BOOT_URL}) : {e!r}", "transient")
	except Exception as e:
		return fail("load_to_db", f"Erreur inattendue : {e!r}")


def analyze_failure(node: str, error: str) -> DebugAnalysis:
	prompt = (
		"Tu es un expert DevOps, Python/LangGraph et backend Java Spring Boot.\n"
		f"L'agent '{node}' d'un pipeline (e-mails de réservation Smoobu -> API Spring Boot -> PostgreSQL) "
		f"a échoué.\n\nErreur :\n{error[:4000]}\n\n"
		"Donne la cause racine et des solutions concrètes (commandes, fichiers, valeurs). "
		"can_auto_fix=True uniquement pour une panne passagère qu'une relance peut résoudre."
	)
	try:
		return structured_llm(DebugAnalysis).invoke(prompt)
	except Exception as e:  # le debug ne doit jamais faire planter le graphe
		return DebugAnalysis(
			root_cause=error[:300], can_auto_fix=False,
			explanation=f"Analyse automatique indisponible ({e!r}).",
			solutions=["Lire le message d'erreur ci-dessus.",
					   "Corriger la cause, puis relancer : python smoobu_pipeline.py --unblock"])


def debug_agent_node(state: AgentState) -> dict:
	print("--- [7] debug_agent : analyse de la panne ---")
	node = state.get("failed_node") or "inconnu"
	error = state.get("current_error") or "erreur inconnue"
	kind = state.get("error_kind", "permanent")
	attempts = dict(state.get("attempts", {}))
	tries = attempts.get(node, 0)

	analysis = analyze_failure(node, error)
	report = analysis.model_dump()

	# Garde-fous : le LLM propose, le code décide.
	can_retry = (analysis.can_auto_fix and node in RETRYABLE_NODES
				 and kind == "transient" and tries < MAX_RETRIES)
	if can_retry:
		attempts[node] = tries + 1
		wait = 5 * (tries + 1)
		print(f"Panne passagère : relance de '{node}' dans {wait}s (essai {tries + 1}/{MAX_RETRIES}).")
		time.sleep(wait)
		return {"attempts": attempts, "requires_human": False, "current_error": None, "debug_report": report}

	print(f"Intervention humaine requise pour '{node}'.")
	return {"attempts": attempts, "requires_human": True, "debug_report": report}


def notifier_node(state: AgentState) -> dict:
	print("--- [8] notifier : e-mail d'alerte et blocage ---")
	node = state.get("failed_node") or "inconnu"
	error = state.get("current_error") or "erreur inconnue"
	rep = state.get("debug_report", {})
	solutions = "\n".join(f"  {i}. {s}" for i, s in enumerate(rep.get("solutions", []), 1)) or "  (aucune proposition)"

	body = (
		f"Bonjour,\n\nLe pipeline Smoobu est BLOQUÉ à l'étape '{node}'.\n\n"
		f"ERREUR\n{error}\n\n"
		f"CAUSE PROBABLE\n{rep.get('root_cause', 'non déterminée')}\n\n"
		f"ANALYSE\n{rep.get('explanation', '')}\n\n"
		f"SOLUTIONS À APPLIQUER\n{solutions}\n\n"
		f"Relances automatiques déjà tentées : {state.get('attempts', {}).get(node, 0)}\n"
		f"Les e-mails de réservation ne sont pas perdus : ils restent non lus dans Gmail.\n"
		f"Après correction, relancez : python smoobu_pipeline.py --unblock\n"
	)
	BLOCK_FILE.write_text(json.dumps({"node": node, "error": error[:2000], "since": time.ctime()}, indent=2),
						  encoding="utf-8")
	try:
		send_mail(f"[ALERTE] Pipeline Smoobu bloqué : '{node}'", body)
		print(f"E-mail envoyé à {ALERT_TO}. Pipeline bloqué ({BLOCK_FILE}).")
		return {"notification_sent": True}
	except Exception as e:
		print(f"Échec de l'envoi de l'e-mail : {e!r}\n\n{body}")
		return {"notification_sent": False}


# --------------------------------------------------------------------------- #
# 6. GRAPHE                                                                   #
# --------------------------------------------------------------------------- #
CHAIN = [
	("code_reviewer", code_reviewer_node),
	("run_test", run_test_node),
	("pipeline_email", pipeline_email_node),
	("data_transformer", data_transformer_node),
	("load_list", load_list_node),
	("load_to_db", load_to_db_node),
]


def route_after_email(state: AgentState) -> str:
	if state.get("current_error"):
		return "debug"
	return "next" if state.get("raw_emails") else "stop"  # rien de nouveau : on s'arrête là


def route_after_debug(state: AgentState) -> str:
	return "notifier" if state.get("requires_human") else state["failed_node"]


def make_router(node_name: str):
	"""Routeur de sortie d'un agent. Retourne 'debug', 'stop' ou 'next'.

	Chaque routeur porte un nom lisible (route_after_<agent>) qui apparaît dans LangSmith.
	"""

	def router(state: AgentState) -> str:
		if state.get("current_error"):
			return "debug"  # l'agent a échoué : le debug prend la main
		key = STOP_IF_EMPTY.get(node_name)
		if key and not state.get(key):
			print(f"Rien à traiter après '{node_name}' : fin du cycle.")
			return "stop"  # rien de nouveau : inutile de continuer
		return "next"  # tout va bien : agent suivant

	router.__name__ = f"route_after_{node_name}"
	return router


STOP_IF_EMPTY = {"pipeline_email": "raw_emails"}


def build_graph():
	wf = StateGraph(AgentState)
	for name, fn in CHAIN:
		wf.add_node(name, fn)
	wf.add_node("debug_agent", debug_agent_node)
	wf.add_node("notifier", notifier_node)
	wf.set_entry_point(CHAIN[0][0])
	#
	# IMPORTANT : un nœud a SOIT des arêtes conditionnelles, SOIT une arête simple, jamais les deux.
	# (avec les deux, LangGraph lance les deux branches : c'est ce qui rendait le graphe illisible)
	for i, (name, _) in enumerate(CHAIN):
		nxt = CHAIN[i + 1][0] if i + 1 < len(CHAIN) else END
		targets = {"debug": "debug_agent", "next": nxt}
		if name in STOP_IF_EMPTY:
			targets["stop"] = END
		wf.add_conditional_edges(name, make_router(name), targets)

	destinations = {"notifier": "notifier", **{name: name for name, _ in CHAIN}}
	wf.add_conditional_edges("debug_agent", route_after_debug, destinations)
	wf.add_edge("notifier", END)
	return wf.compile()


app = build_graph()  # exposé pour LangGraph Studio (langgraph.json -> smoobu_pipeline.py:app)


# --------------------------------------------------------------------------- #
# 7. EXÉCUTION                                                                #
# --------------------------------------------------------------------------- #
def check_langsmith() -> None:
	if not LANGSMITH_ON:
		print("LangSmith DÉSACTIVÉ : aucune clé (LANGSMITH_API_KEY ou LANGCHAIN_API_KEY) trouvée.")
		return
	try:
		from langsmith import Client
		client = Client()
		list(client.list_projects(limit=1))
		print(f"LangSmith OK : endpoint {client.api_url}, projet '{os.environ['LANGSMITH_PROJECT']}'.")
	except Exception as e:
		print(f"LangSmith ERREUR : {e!r}\n"
			  "-> Vérifiez la clé. Si votre compte est en Europe, définissez\n"
			  "   LANGSMITH_ENDPOINT=https://eu.api.smith.langchain.com")


def flush_traces() -> None:
	"""Le script s'arrête vite : sans ceci, les traces partent en tâche de fond et sont perdues."""
	try:
		from langchain_core.tracers.langchain import wait_for_all_tracers
		wait_for_all_tracers()
	except Exception as e:
		print(f"Envoi des traces LangSmith incomplet : {e!r}")


def main() -> int:
	parser = argparse.ArgumentParser()
	parser.add_argument("--graph", action="store_true", help="affiche le graphe Mermaid")
	parser.add_argument("--unblock", action="store_true", help="lève le blocage du pipeline")
	args = parser.parse_args()

	if args.graph:
		print(app.get_graph().draw_mermaid())
		return 0
	if args.unblock:
		BLOCK_FILE.unlink(missing_ok=True)
		print("Pipeline débloqué.")
		return 0

	missing = [v for v in ("ANTHROPIC_API_KEY", "GMAIL_APP_PASSWORD") if not os.getenv(v)]
	if missing:
		print(f"Variables d'environnement manquantes : {', '.join(missing)}")
		return 1
	if BLOCK_FILE.exists():
		print(f"Pipeline BLOQUÉ ({BLOCK_FILE.read_text(encoding='utf-8')}).\n"
			  "Corrigez le problème puis lancez : python smoobu_pipeline.py --unblock")
		return 2

	print("=== Démarrage du pipeline Smoobu multi-agents ===")
	check_langsmith()
	config = {
		"run_name": "smoobu_pipeline",
		"tags": ["smoobu", "multi-agents"],
		"metadata": {"model": MODEL, "backend": SPRING_BOOT_URL},
		"recursion_limit": 40,
	}
	try:
		final = app.invoke({"attempts": {}, "requires_human": False, "current_error": None,
							"failed_node": None}, config=config)
	finally:
		flush_traces()

	if final.get("requires_human"):
		print("\n=== Pipeline BLOQUÉ : intervention humaine requise ===")
		return 2
	print("\n=== Cycle terminé sans erreur ===")
	return 0


if __name__ == "__main__":
	sys.exit(main())
