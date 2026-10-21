"""
Synthetic Support Case Generator
--------------------------------
Generates a realistic, reproducible dataset of 200 customer support cases across 5 departments:
- Billing
- Technical
- Sales
- Operations
- Other

Includes urgency levels, churn risk flags, and multilingual test splits (English, Spanish, Hindi).
Split: 140 train, 30 validation, 30 test.
"""

import os
import random
import pandas as pd
import numpy as np

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

DEPARTMENTS = ["Billing", "Technical", "Sales", "Operations", "Other"]
URGENCIES = ["Low", "Medium", "Critical"]

# Curated realistic templates per department
CASE_TEMPLATES = {
    "Billing": [
        ("Discrepancy on invoice #{num}", "We noticed an unexpected charge of ${val} on invoice #{num} for our enterprise subscription. Please clarify before we authorize payment.", "Medium", "No"),
        ("Overcharged for inactive seats", "Our account was billed for 45 seats this month even though 10 were de-provisioned last month. We need an immediate refund credit.", "Medium", "Yes"),
        ("Urgent: Payment gateway failure & subscription cancelled", "Our automatic credit card renewal failed with error 402, and your system locked out 80 agents! Restore access immediately.", "Critical", "Yes"),
        ("Requesting VAT tax exemption update", "Attached is our updated W-9 and EU VAT exemption certificate. Please regenerate our Q3 invoice without tax.", "Low", "No"),
        ("Annual contract billing schedule inquiry", "Could you provide the payment schedule and wire transfer instructions for our upcoming annual renewal?", "Low", "No"),
        ("Double billing detected on corporate card", "Our finance department flagged two identical charges of ${val} posted on the same day. Please reverse one.", "Critical", "Yes"),
        ("Past due notice received in error", "We received an automated dunning email stating invoice #{num} is overdue, but wire confirmation shows it cleared on the 12th.", "Medium", "No"),
        ("Update billing contact and PO number", "Please update the billing email to accounts-payable@company.com and reference PO-98412 on all future statements.", "Low", "No"),
    ],
    "Technical": [
        ("REST API 500 error on webhook dispatch", "Since 09:00 UTC, all outbound webhook events to our ingestion endpoint fail with HTTP 500 Internal Server Error. Logs attached.", "Critical", "Yes"),
        ("SSO SAML authentication loop with Okta", "Our engineering team cannot log into the dashboard via Okta SAML. Users are caught in an infinite redirect loop.", "Critical", "Yes"),
        ("Query latency spike on GraphQL endpoint", "Our reporting queries that usually take 200ms are now timing out at 30s. Has there been a database migration or index drop?", "Medium", "No"),
        ("SDK TypeError: undefined is not an object", "When calling client.cases.create() in the Python SDK v2.4, we get an unhandled exception when passing custom metadata.", "Medium", "No"),
        ("Export to CSV contains corrupted UTF-8 characters", "When exporting daily audit logs containing Japanese characters, the resulting CSV displays gibberish question marks.", "Low", "No"),
        ("Rate limit reached unexpectedly on Sandbox", "Our CI pipeline is failing because API calls return 429 Too Many Requests, even though rate limits were supposed to be lifted.", "Medium", "No"),
        ("WebSocket connection dropping every 60 seconds", "Persistent live-feed WebSocket disconnects with code 1006 every minute. Keep-alive pings seem unacknowledged.", "Medium", "No"),
        ("Data synchronization lag in replica database", "Secondary read replicas are lagging by over 45 minutes behind primary, causing stale read bugs on our user portal.", "Critical", "Yes"),
    ],
    "Sales": [
        ("Upgrading to Enterprise Tier - custom quote", "Our startup just closed Series B and we need to upgrade from Growth to Enterprise with 150 dedicated seats. Need pricing.", "Medium", "No"),
        ("Contract renewal discount discussion", "Our 1-year agreement expires next month. We're evaluating alternatives unless we can discuss competitive volume pricing.", "Critical", "Yes"),
        ("Request for product demo - Field Service add-on", "Our operations VP would like a 30-minute demo of the new Field Service dispatcher capability this Thursday.", "Low", "No"),
        ("Add 25 additional agent licenses", "We hired a new support cohort starting Monday and need to provision 25 additional licenses under our master MSA.", "Medium", "No"),
        ("Competitor price matching inquiry", "We received a quote from Zendesk at 20% lower cost per seat. Can your commercial team match this to keep our business?", "Critical", "Yes"),
        ("Custom SLA terms for dedicated hosting", "Our compliance team requires 99.99% uptime with financial penalty clauses. Can we negotiate custom SLA terms?", "Medium", "No"),
        ("Trial extension request for POC evaluation", "Our security audit took longer than expected. Can we extend our enterprise trial sandbox for another 14 days?", "Low", "No"),
        ("Partnership reseller inquiry", "We are an AWS Premier Consulting partner interested in joining your reseller program to bundle your platform.", "Low", "No"),
    ],
    "Operations": [
        ("Shipment tracking delay for hardware dispatch", "The replacement scanner hardware scheduled for delivery on Monday is still marked as pending in warehouse status.", "Medium", "No"),
        ("Scheduled maintenance notification clash", "Your scheduled maintenance window on Saturday 2 AM overlaps with our peak Black Friday deployment. Can we reschedule?", "Critical", "Yes"),
        ("Vendor compliance questionnaire completion", "Please have your InfoSec operations team complete the attached SOC 2 Type II vendor assessment spreadsheet.", "Low", "No"),
        ("Office physical badge provisioning delay", "Contractor technician badges requested 5 days ago for the Chicago data center have not been cleared by security ops.", "Low", "No"),
        ("Supply chain delay for replacement parts", "We have 12 field units stalled awaiting shipment of motherboard replacement modules part #ET-882.", "Critical", "Yes"),
        ("Onboarding workflow stuck in pending approval", "New employee provisioning request #OP-4410 has been stalled at manager sign-off for 4 business days.", "Medium", "No"),
        ("Facility cooling outage alert followup", "Root cause inquiry regarding the auxiliary HVAC failover incident in zone 3 reported yesterday afternoon.", "Medium", "No"),
        ("Courier damaged packaging on arrival", "Batch parcel arrived with crushed exterior packaging. Inspection required before inventory check-in.", "Low", "No"),
    ],
    "Other": [
        ("General inquiry regarding public API roadmap", "Are there plans to support gRPC streaming in the Q4 public release? We could not find mentions on your blog.", "Low", "No"),
        ("GDPR data subject deletion request (RTBF)", "Pursuant to Article 17 GDPR, please permanently erase all personal telemetry records associated with user ID 9104.", "Medium", "No"),
        ("Feedback on recent UI navigation redesign", "The recent dashboard update moved the filter controls behind a hamburger menu which adds two unnecessary clicks.", "Low", "No"),
        ("Speaker invitation for Annual Tech Summit", "We would love to invite your VP of AI to deliver a keynote at our upcoming Distributed Systems summit in Austin.", "Low", "No"),
        ("Career inquiry: Staff Infrastructure Engineer", "I submitted an application for the remote platform role and wanted to verify receipt of my portfolio link.", "Low", "No"),
        ("Copyright DMCA notice acknowledgment", "Formal confirmation required regarding the takedown notice processed for community forum thread #4419.", "Medium", "No"),
        ("Lost and found inquiry from visitor center", "Checking if an encrypted laptop charger left in conference room B on Tuesday was turned in to security.", "Low", "No"),
        ("Unsubscribe from marketing email updates", "Please remove our corporate domain from all promotional webinars while keeping transactional alerts active.", "Low", "No"),
    ]
}

# Translations for test set (Spanish and Hindi)
SPANISH_TRANSLATIONS = {
    "Billing": [
        ("Discrepancia en la factura #{num}", "Notamos un cargo inesperado de ${val} en la factura #{num} de nuestra suscripción empresarial. Por favor aclarar antes del pago.", "Medium", "No"),
        ("Cobro excesivo por asientos inactivos", "Se nos facturaron 45 puestos este mes aunque 10 fueron dados de baja. Necesitamos un reembolso inmediato.", "Medium", "Yes"),
        ("Urgente: Falla de pasarela de pago y servicio cancelado", "La renovación automática con tarjeta falló con error 402 y bloquearon a 80 agentes. Restauren el acceso de inmediato.", "Critical", "Yes"),
        ("Actualización de exención de impuestos", "Adjunto nuestro certificado de exención fiscal actualizado. Por favor regenerar la factura sin impuestos.", "Low", "No"),
        ("Consulta sobre calendario de facturación anual", "¿Podrían proporcionarnos el calendario de pagos y las instrucciones para la transferencia bancaria anual?", "Low", "No"),
        ("Doble cargo detectado en tarjeta corporativa", "Contabilidad detectó dos cargos idénticos de ${val} el mismo día. Por favor reembolsar uno de inmediato.", "Critical", "Yes"),
    ],
    "Technical": [
        ("Error 500 de API REST en envío de webhook", "Desde las 09:00 UTC, todos los webhooks a nuestro endpoint fallan con HTTP 500 Error Interno. Adjunto logs.", "Critical", "Yes"),
        ("Bucle de autenticación SSO SAML con Okta", "Los ingenieros no pueden acceder al panel mediante Okta SAML. Los usuarios quedan atrapados en bucle infinito.", "Critical", "Yes"),
        ("Pico de latencia en consultas GraphQL", "Las consultas de reportes que tomaban 200ms ahora fallan por timeout de 30s. ¿Hubo migración de base de datos?", "Medium", "No"),
        ("TypeError en SDK: undefined is not an object", "Al invocar client.cases.create() en Python SDK v2.4 recibimos una excepción no controlada.", "Medium", "No"),
        ("Caracteres corruptos al exportar a CSV", "Al exportar registros de auditoría que contienen caracteres especiales, el archivo muestra signos de interrogación.", "Low", "No"),
        ("Límite de tasa excedido inesperadamente", "Las pruebas automáticas fallan con error 429 Too Many Requests a pesar de que el límite debía ser ilimitado.", "Medium", "No"),
    ],
    "Sales": [
        ("Actualización a plan Enterprise - cotización", "Acabamos de cerrar ronda de inversión y necesitamos actualizar a plan Enterprise para 150 puestos. Solicitamos precios.", "Medium", "No"),
        ("Discusión de descuento para renovación de contrato", "Nuestro contrato anual expira el próximo mes. Estamos evaluando alternativas a menos que ofrezcan descuento.", "Critical", "Yes"),
        ("Solicitud de demostración de producto", "A nuestro director le gustaría una demo de 30 minutos de la nueva funcionalidad este jueves.", "Low", "No"),
        ("Agregar 25 licencias de usuario adicionales", "Contratamos un nuevo equipo que inicia el lunes y requerimos activar 25 licencias bajo nuestro acuerdo marco.", "Medium", "No"),
        ("Igualación de precio de la competencia", "Recibimos cotización de Zendesk con un costo 20% menor por puesto. ¿Pueden igualar esta oferta?", "Critical", "Yes"),
        ("Términos de SLA personalizados", "Requerimos 99.99% de disponibilidad con cláusulas de penalización financiera. ¿Podemos negociar términos?", "Medium", "No"),
    ],
    "Operations": [
        ("Retraso en seguimiento de envío de hardware", "El hardware de reemplazo programado para entrega el lunes aún figura como pendiente en el almacén.", "Medium", "No"),
        ("Conflicto con horario de mantenimiento programado", "El mantenimiento del sábado a las 2 AM coincide con nuestro pico de tráfico. ¿Se puede reprogramar?", "Critical", "Yes"),
        ("Cuestionario de cumplimiento de seguridad", "Por favor soliciten al equipo de operaciones completar el cuestionario de evaluación SOC 2 adjunto.", "Low", "No"),
        ("Retraso en credenciales de acceso a instalaciones", "Los pases solicitados para técnicos hace 5 días no han sido aprobados por seguridad física.", "Low", "No"),
        ("Demora en cadena de suministro de piezas", "Tenemos 12 unidades en campo detenidas esperando el envío de placas madre de repuesto.", "Critical", "Yes"),
        ("Flujo de aprobación bloqueado", "La solicitud de aprovisionamiento #OP-4410 lleva 4 días hábiles sin firma del supervisor.", "Medium", "No"),
    ],
    "Other": [
        ("Consulta general sobre hoja de ruta de API pública", "¿Existen planes para admitir gRPC streaming en el lanzamiento del cuarto trimestre?", "Low", "No"),
        ("Solicitud de eliminación de datos personales (GDPR)", "Conforme al Artículo 17 del RGPD, solicitamos eliminar permanentemente los registros del usuario 9104.", "Medium", "No"),
        ("Comentarios sobre el rediseño de interfaz", "La actualización reciente movió los filtros dentro de un menú adicional, lo que genera clics innecesarios.", "Low", "No"),
        ("Invitación a conferencia anual de tecnología", "Nos gustaría invitar a su director de IA a dar una ponencia magistral en nuestra cumbre de Austin.", "Low", "No"),
        ("Consulta sobre postulación laboral", "Envié mi solicitud para el puesto de ingeniero de infraestructura y deseo confirmar la recepción del portafolio.", "Low", "No"),
        ("Confirmación de aviso de derechos de autor", "Requerimos confirmación formal del procesamiento del aviso de retiro del foro comunitario.", "Medium", "No"),
    ]
}

HINDI_TRANSLATIONS = {
    "Billing": [
        ("चालान #{num} में विसंगति", "हमने अपने सदस्यता चालान #{num} पर ${val} का अप्रत्याशित शुल्क देखा है। भुगतान से पहले कृपया स्पष्ट करें।", "Medium", "No"),
        ("निष्क्रिय सीटों के लिए अतिरिक्त शुल्क", "इस महीने 45 सीटों का बिल भेजा गया जबकि 10 सीटें हटा दी गई थीं। हमें तत्काल धनवापसी चाहिए।", "Medium", "Yes"),
        ("अत्यावश्यक: भुगतान विफलता और सदस्यता रद्द", "क्रेडिट कार्ड नवीनीकरण विफल हो गया और सिस्टम ने 80 एजेंटों को लॉक कर दिया! तुरंत एक्सेस बहाल करें।", "Critical", "Yes"),
        ("कर छूट प्रमाणपत्र अद्यतन", "संलग्न हमारा कर छूट प्रमाणपत्र है। कृपया बिना कर के नया चालान जारी करें।", "Low", "No"),
        ("वार्षिक अनुबंध बिलिंग अनुसूची पूछताछ", "क्या आप हमारे आगामी वार्षिक नवीनीकरण के लिए भुगतान अनुसूची और बैंक विवरण प्रदान कर सकते हैं?", "Low", "No"),
        ("कॉर्पोरेट कार्ड पर दोहरा शुल्क", "वित्त विभाग ने एक ही दिन में ${val} के दो समान शुल्क दर्ज किए हैं। कृपया एक वापस करें।", "Critical", "Yes"),
    ],
    "Technical": [
        ("वेबहूक प्रेषण पर REST API 500 त्रुटि", "सुबह 09:00 UTC से, हमारे समापन बिंदु पर सभी आउटबाउंड वेबहूक HTTP 500 आंतरिक त्रुटि के साथ विफल हो रहे हैं।", "Critical", "Yes"),
        ("Okta के साथ SSO SAML प्रमाणीकरण लूप", "इंजीनियर Okta SAML के माध्यम से लॉग इन नहीं कर पा रहे हैं। उपयोगकर्ता अनंत रीडायरेक्ट में फंस गए हैं।", "Critical", "Yes"),
        ("GraphQL समापन बिंदु पर विलंबता वृद्धि", "रिपोर्टिंग क्वेरी जो 200ms लेती थीं अब 30s टाइमआउट हो रही हैं। क्या डेटाबेस माइग्रेशन हुआ है?", "Medium", "No"),
        ("SDK त्रुटि: undefined is not an object", "Python SDK v2.4 में client.cases.create() कॉल करते समय अनपेक्षित अपवाद प्राप्त होता है।", "Medium", "No"),
        ("CSV निर्यात में दूषित वर्ण", "ऑडिट लॉग निर्यात करते समय उत्पन्न CSV फ़ाइल में अस्पष्ट प्रश्न चिह्न दिखाई देते हैं।", "Low", "No"),
        ("सैंडबॉक्स पर दर सीमा पार हो गई", "हमारी CI पाइपलाइन विफल हो रही है क्योंकि API 429 Too Many Requests लौटा रहा है।", "Medium", "No"),
    ],
    "Sales": [
        ("एंटरप्राइज टियर में अपग्रेड - कस्टम कोट", "हमें 150 सीटों के लिए एंटरप्राइज में अपग्रेड करना है। कृपया मूल्य निर्धारण प्रदान करें।", "Medium", "No"),
        ("अनुबंध नवीनीकरण छूट पर चर्चा", "हमारा अनुबंध अगले महीने समाप्त हो रहा है। यदि छूट नहीं मिली तो हम अन्य विकल्पों पर विचार कर रहे हैं।", "Critical", "Yes"),
        ("उत्पाद डेमो अनुरोध", "हमारे उपाध्यक्ष इस गुरुवार को फील्ड सर्विस सुविधा का 30 मिनट का डेमो देखना चाहते हैं।", "Low", "No"),
        ("25 अतिरिक्त लाइसेंस जोड़ें", "हमारी नई टीम सोमवार से शुरू हो रही है और हमें 25 अतिरिक्त लाइसेंस सक्रिय करने की आवश्यकता है।", "Medium", "No"),
        ("प्रतिस्पर्धी मूल्य मिलान पूछताछ", "हमें प्रतिस्पर्धी से 20% कम कीमत का प्रस्ताव मिला है। क्या आपकी टीम इसका मिलान कर सकती है?", "Critical", "Yes"),
        ("समर्पित होस्टिंग के लिए कस्टम SLA शर्तें", "हमारी अनुपालन टीम को 99.99% अपटाइम की आवश्यकता है। क्या हम शर्तों पर बातचीत कर सकते हैं?", "Medium", "No"),
    ],
    "Operations": [
        ("हार्डवेयर प्रेषण के लिए ट्रैकिंग में देरी", "सोमवार को आने वाला प्रतिस्थापन हार्डवेयर अभी भी गोदाम की स्थिति में लंबित दिख रहा है।", "Medium", "No"),
        ("निर्धारित रखरखाव सूचना टकराव", "शनिवार को निर्धारित रखरखाव हमारी व्यस्त बिक्री अवधि के साथ टकराता है। क्या इसे बदला जा सकता है?", "Critical", "Yes"),
        ("विक्रेता अनुपालन प्रश्नावली", "कृपया अपनी सुरक्षा संचालन टीम से संलग्न SOC 2 मूल्यांकन स्प्रेडशीट भरने को कहें।", "Low", "No"),
        ("कार्यालय प्रवेश बैज में देरी", "डेटा सेंटर के लिए 5 दिन पहले मांगे गए सुरक्षा बैज अभी तक जारी नहीं किए गए हैं।", "Low", "No"),
        ("पुर्जों की आपूर्ति श्रृंखला में देरी", "मदरबोर्ड मॉड्यूल के शिपमेंट की प्रतीक्षा में हमारी 12 फील्ड इकाइयां रुकी हुई हैं।", "Critical", "Yes"),
        ("ऑनबोर्डिंग प्रक्रिया अनुमोदन में अटकी", "कर्मचारी प्रावधान अनुरोध 4 व्यावसायिक दिनों से प्रबंधक के हस्ताक्षर के बिना अटका है।", "Medium", "No"),
    ],
    "Other": [
        ("सार्वजनिक API रोडमैप के संबंध में सामान्य पूछताछ", "क्या चौथी तिमाही में gRPC स्ट्रीमिंग का समर्थन करने की योजना है?", "Low", "No"),
        ("व्यक्तिगत डेटा हटाने का अनुरोध (GDPR)", "GDPR अनुच्छेद 17 के तहत, कृपया उपयोगकर्ता आईडी 9104 से संबंधित सभी रिकॉर्ड स्थायी रूप से हटा दें।", "Medium", "No"),
        ("UI नेविगेशन पुनर्रचना पर प्रतिक्रिया", "हालिया अपडेट ने फ़िल्टर नियंत्रणों को मेनू के पीछे ले जाया, जिससे अतिरिक्त क्लिक बढ़ गए हैं।", "Low", "No"),
        ("वार्षिक टेक शिखर सम्मेलन के लिए मुख्य वक्ता आमंत्रण", "हम आपके AI निदेशक को ऑस्टिन में आगामी शिखर सम्मेलन में मुख्य भाषण देने के लिए आमंत्रित करना चाहते हैं।", "Low", "No"),
        ("करियर पूछताछ: अवसंरचना इंजीनियर", "मैंने प्लेटफॉर्म भूमिका के लिए आवेदन जमा किया था और पोर्टफोलियो प्राप्ति की पुष्टि चाहता हूं।", "Low", "No"),
        ("कॉपीराइट सूचना पावती", "मंच थ्रेड के संबंध में संसाधित टेकडाउन नोटिस के लिए औपचारिक पुष्टि आवश्यक है।", "Medium", "No"),
    ]
}


def generate_dataset():
    cases = []
    case_counter = 1000

    # Generate 200 cases across 5 departments (40 cases per department)
    for dept in DEPARTMENTS:
        templates = CASE_TEMPLATES[dept]
        for i in range(40):
            case_counter += 1
            tpl_subj, tpl_body, base_urgency, base_churn = templates[i % len(templates)]
            
            # Param substitutions
            inv_num = random.randint(10450, 99820)
            val = random.choice([249, 499, 1250, 3400, 8900, 15000])
            
            subject = tpl_subj.replace("{num}", str(inv_num)).replace("{val}", str(val))
            body = tpl_body.replace("{num}", str(inv_num)).replace("{val}", str(val))
            
            # Subtle realistic noise & variations
            if i % 3 == 0:
                body += f" Reference ID: REF-{random.randint(100, 999)}."
            elif i % 5 == 0:
                body += " Please escalate to a tier-2 specialist as soon as possible."

            cases.append({
                "case_id": f"CS-{case_counter}",
                "subject": subject,
                "body": body,
                "department": dept,
                "urgency": base_urgency,
                "churn_risk": base_churn,
                "language": "English",
            })

    # Shuffle deterministically
    random.shuffle(cases)
    df = pd.DataFrame(cases)

    # 140 train, 30 val, 30 test
    train_df = df.iloc[:140].copy().reset_index(drop=True)
    val_df = df.iloc[140:170].copy().reset_index(drop=True)
    test_df = df.iloc[170:200].copy().reset_index(drop=True)

    # Save to disk
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, "data")
    splits_dir = os.path.join(data_dir, "splits")
    os.makedirs(splits_dir, exist_ok=True)

    df.to_csv(os.path.join(data_dir, "synthetic_cases.csv"), index=False)
    train_df.to_csv(os.path.join(splits_dir, "train.csv"), index=False)
    val_df.to_csv(os.path.join(splits_dir, "val.csv"), index=False)
    test_df.to_csv(os.path.join(splits_dir, "test.csv"), index=False)

    # Generate exact paired test sets in Spanish and Hindi for the 30 test cases
    test_es = []
    test_hi = []
    for idx, row in test_df.iterrows():
        dept = row["department"]
        es_pool = SPANISH_TRANSLATIONS[dept]
        hi_pool = HINDI_TRANSLATIONS[dept]
        match_idx = idx % len(es_pool)
        
        inv_num = random.randint(10450, 99820)
        val = random.choice([249, 499, 1250, 3400, 8900])

        es_subj, es_body, es_urg, es_churn = es_pool[match_idx]
        hi_subj, hi_body, hi_urg, hi_churn = hi_pool[match_idx]

        test_es.append({
            "case_id": f"{row['case_id']}-ES",
            "subject": es_subj.replace("{num}", str(inv_num)).replace("{val}", str(val)),
            "body": es_body.replace("{num}", str(inv_num)).replace("{val}", str(val)),
            "department": dept,
            "urgency": es_urg,
            "churn_risk": es_churn,
            "language": "Spanish",
        })

        test_hi.append({
            "case_id": f"{row['case_id']}-HI",
            "subject": hi_subj.replace("{num}", str(inv_num)).replace("{val}", str(val)),
            "body": hi_body.replace("{num}", str(inv_num)).replace("{val}", str(val)),
            "department": dept,
            "urgency": hi_urg,
            "churn_risk": hi_churn,
            "language": "Hindi",
        })

    pd.DataFrame(test_es).to_csv(os.path.join(splits_dir, "test_spanish.csv"), index=False)
    pd.DataFrame(test_hi).to_csv(os.path.join(splits_dir, "test_hindi.csv"), index=False)

    print(f"✅ Generated 200 synthetic cases successfully!")
    print(f"   Train: {len(train_df)} cases (saved to data/splits/train.csv)")
    print(f"   Val:   {len(val_df)} cases (saved to data/splits/val.csv)")
    print(f"   Test:  {len(test_df)} cases (saved to data/splits/test.csv)")
    print(f"   Spanish Test: {len(test_es)} cases (saved to data/splits/test_spanish.csv)")
    print(f"   Hindi Test:   {len(test_hi)} cases (saved to data/splits/test_hindi.csv)")

if __name__ == "__main__":
    generate_dataset()
