import smoobu_pipeline as sp

AIRBNB_EMAIL = """Neue Buchung: Loft am Pegnitztal Nürnberg: 29.10.26 – 02.11.26
Du hast eine neue Buchung!
Anreise: Do., 29.10.26
Abreise: Mo., 02.11.26
Gast: UAB Signallink Mikalajunas
Gastnachricht: Buchungsnummer: HMC8Q9NJTW
"""


def _extract(raw):
	result = sp.data_transformer_node({"raw_emails": [raw]})
	assert result["current_error"] is None, result["current_error"]
	return sp.load_list_node({"transformed_data": result["transformed_data"]})["batched_payloads"][0][0]


def test_sample_email_gives_backend_json():
	booking = _extract(sp.SAMPLE_EMAIL)
	assert set(booking) == {"propertyName", "checkIn", "checkOut", "guestName", "bookingNumber", "guestMessage"}
	assert booking["checkIn"] == "2026-11-06"
	assert booking["checkOut"] == "2026-11-08"
	assert booking["bookingNumber"] == "6041336634"


def test_airbnb_booking_number_with_letters():
	booking = _extract(AIRBNB_EMAIL)
	assert booking["bookingNumber"] == "HMC8Q9NJTW"
	assert booking["guestMessage"] is None


def test_html_only_email_is_read():
	from email.message import EmailMessage
	msg = EmailMessage()
	msg.set_content("<p>Anreise: Do., 29.10.26</p><p>Gast: Test</p>", subtype="html")
	text = sp._plain_text(msg)
	assert "Anreise: Do., 29.10.26" in text
	assert "Gast: Test" in text


def test_email_without_booking_fields_fails():
	result = sp.data_transformer_node({"raw_emails": ["*testtt*"]})
	assert result["failed_node"] == "data_transformer"


def test_scheduled_graph_skips_review_and_tests():
	nodes = set(sp.build_graph(sp.SCHEDULED_CHAIN).get_graph().nodes)
	assert "code_reviewer" not in nodes and "run_test" not in nodes
	assert "pipeline_email" in nodes
