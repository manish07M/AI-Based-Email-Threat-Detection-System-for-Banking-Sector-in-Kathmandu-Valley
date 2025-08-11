from flask import Flask, request, render_template_string
import re
import pickle
import pandas as pd
from urllib.parse import urlparse
import requests
from bs4 import BeautifulSoup
from datetime import datetime
import math
import hashlib

app = Flask(__name__)

# Load models
with open('url_threat_xgb_model.pkl', 'rb') as f:
    xgb_model = pickle.load(f)
with open('url_threat_lgb_model.pkl', 'rb') as f:
    lgb_model = pickle.load(f)
with open('url_anomaly_model.pkl', 'rb') as f:
    anomaly_model = pickle.load(f)
with open('malware_model.pkl', 'rb') as f:
    malware_model = pickle.load(f)
with open('spam_model.pkl', 'rb') as f:
    spam_model = pickle.load(f)
with open('attachment_malware_model.pkl', 'rb') as f:
    attachment_malware_model = pickle.load(f)

# In-memory history
email_history = []

# HTML Template with blue gradient styling
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Phishing, Malware, and Spam Detection</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            background: linear-gradient(to right, #1e3a8a, #60a5fa);
            color: #1e3a8a;
            margin: 0;
            padding: 20px;
            display: flex;
            justify-content: center;
            align-items: center;
            min-height: 100vh;
        }
        .container {
            background: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
            max-width: 800px;
            width: 100%;
        }
        h1 {
            color: #1e3a8a;
            text-align: center;
        }
        textarea, input[type="text"] {
            width: 100%;
            padding: 10px;
            margin: 10px 0;
            border: 1px solid #60a5fa;
            border-radius: 5px;
        }
        input[type="submit"] {
            background: #1e3a8a;
            color: white;
            padding: 10px 20px;
            border: none;
            border-radius: 5px;
            cursor: pointer;
        }
        input[type="submit"]:hover {
            background: #3b82f6;
        }
        .result, .report {
            margin-top: 20px;
            padding: 15px;
            background: #dbeafe;
            border-radius: 5px;
        }
        .history {
            margin-top: 20px;
        }
        .history table {
            width: 100%;
            border-collapse: collapse;
            margin-top: 10px;
        }
        .history th, .history td {
            border: 1px solid #60a5fa;
            padding: 8px;
            text-align: left;
        }
        .history th {
            background: #1e3a8a;
            color: white;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>Phishing, Malware, and Spam Detection</h1>
        <form method="post">
            <textarea name="email_body" rows="5" placeholder="Enter email body with URLs"></textarea>
            <input type="text" name="attachment_info" placeholder="Enter attachment info (e.g., filename.pdf, 500KB)">
            <input type="submit" value="Analyze">
        </form>
        {% if result %}
        <div class="result">
            <h2>Result</h2>
            <p><strong>URL:</strong> {{ url }}</p>
            <p><strong>XGBoost Prediction:</strong> {{ result }}</p>
            <p><strong>LightGBM Prediction:</strong> {{ lgb_result }}</p>
            <p><strong>Anomaly Detection:</strong> {{ anomaly_result }}</p>
            <p><strong>Malware Detection:</strong> {{ malware_result }}</p>
            <p><strong>Spam Detection:</strong> {{ spam_result }}</p>
            {% if attachment_info %}
            <p><strong>Attachment:</strong> {{ attachment_info }}</p>
            <p><strong>Attachment Malware Detection:</strong> {{ attachment_malware_result }}</p>
            {% endif %}
        </div>
        <div class="report">
            {{ report|safe }}
        </div>
        {% endif %}
        <div class="history">
            <h2>Analysis History</h2>
            <table>
                <tr>
                    <th>Timestamp</th>
                    <th>Email Snippet</th>
                    <th>URL</th>
                    <th>XGBoost Result</th>
                    <th>LightGBM Result</th>
                    <th>Anomaly Result</th>
                    <th>Malware Result</th>
                    <th>Spam Result</th>
                    <th>Attachment</th>
                    <th>Attachment Malware</th>
                </tr>
                {% for entry in history %}
                <tr>
                    <td>{{ entry.timestamp }}</td>
                    <td>{{ entry.email_snippet }}</td>
                    <td>{{ entry.url }}</td>
                    <td>{{ entry.result }}</td>
                    <td>{{ entry.lgb_result }}</td>
                    <td>{{ entry.anomaly_result }}</td>
                    <td>{{ entry.malware_result }}</td>
                    <td>{{ entry.spam_result }}</td>
                    <td>{{ entry.attachment_info }}</td>
                    <td>{{ entry.attachment_malware_result }}</td>
                </tr>
                {% endfor %}
            </table>
        </div>
    </div>
</body>
</html>
"""

# URL extraction function
def extract_urls(text):
    url_pattern = r'https?://[^\s<>"]+|www\.[^\s<>"]+'
    return re.findall(url_pattern, text)

# Compute Shannon entropy for a string
def compute_entropy(text):
    if not text:
        return 0.0
    entropy = 0
    for x in set(text):
        p_x = float(text.count(x)) / len(text)
        entropy -= p_x * math.log2(p_x)
    return entropy

# Enhanced feature extraction with web scraping and malware/spam features
def compute_web_features(url):
    try:
        response = requests.get(url, timeout=5)
        soup = BeautifulSoup(response.text, 'html.parser')
        links = soup.find_all('a', href=True)
        external_links = [link['href'] for link in links if urlparse(link['href']).hostname != urlparse(url).hostname]
        features = {
            'PctExtHyperlinks': len(external_links) / len(links) if links else 0.0,
            'ExtFavicon': 1 if soup.find('link', rel='icon', href=lambda h: h and urlparse(h).hostname != urlparse(url).hostname) else 0,
            'InsecureForms': 1 if soup.find('form', action=lambda a: a and not a.startswith('https')) else 0,
        }
        return features
    except:
        return {'PctExtHyperlinks': 0.0, 'ExtFavicon': 0, 'InsecureForms': 0}

def compute_url_features(url, email_body="", include_extended_features=False):
    parsed_url = urlparse(url)
    hostname = parsed_url.hostname or ''
    path = parsed_url.path
    query = parsed_url.query
    web_features = compute_web_features(url)
    features = {
        'NumDots': hostname.count('.'),
        'SubdomainLevel': len([s for s in hostname.split('.') if s]) - 2 if hostname else 0,
        'PathLevel': len([p for p in path.split('/') if p]),
        'UrlLength': len(url),
        'NumDash': url.count('-'),
        'NumDashInHostname': hostname.count('-'),
        'AtSymbol': 1 if '@' in url else 0,
        'TildeSymbol': 1 if '~' in url else 0,
        'NumUnderscore': url.count('_'),
        'NumPercent': url.count('%'),
        'NumQueryComponents': len(query.split('&')) if query else 0,
        'NumAmpersand': url.count('&'),
        'NumHash': url.count('#'),
        'NumNumericChars': sum(c.isdigit() for c in url),
        'NoHttps': 0 if url.startswith('https') else 1,
        'RandomString': 1 if any(c.isalpha() for c in hostname) and len(hostname) > 20 else 0,
        'IpAddress': 1 if re.match(r'^\d+\.\d+\.\d+\.\d+$', hostname) else 0,
        'DomainInSubdomains': 1 if 'domain' in hostname.lower() else 0,
        'DomainInPaths': 1 if 'domain' in path.lower() else 0,
        'HttpsInHostname': 1 if 'https' in hostname.lower() else 0,
        'HostnameLength': len(hostname),
        'PathLength': len(path),
        'QueryLength': len(query),
        'DoubleSlashInPath': 1 if '//' in path else 0,
        'NumSensitiveWords': sum(1 for word in ['login', 'secure', 'account', 'bank'] if word in url.lower()),
        'EmbeddedBrandName': 1 if any(brand in hostname.lower() for brand in ['paypal', 'google', 'facebook']) else 0,
        'PctExtHyperlinks': web_features['PctExtHyperlinks'],
        'PctExtResourceUrls': 0.0,
        'ExtFavicon': web_features['ExtFavicon'],
        'InsecureForms': web_features['InsecureForms'],
        'RelativeFormAction': 0,
        'ExtFormAction': 0,
        'AbnormalFormAction': 0,
        'PctNullSelfRedirectHyperlinks': 0.0,
        'FrequentDomainNameMismatch': 0,
        'FakeLinkInStatusBar': 0,
        'RightClickDisabled': 0,
        'PopUpWindow': 0,
        'SubmitInfoToEmail': 0,
        'IframeOrFrame': 0,
        'MissingTitle': 0,
        'ImagesOnlyInForm': 0,
        'SubdomainLevelRT': 1 if len([s for s in hostname.split('.') if s]) - 2 <= 1 else 0,
        'UrlLengthRT': 1 if len(url) <= 75 else 0,
        'PctExtResourceUrlsRT': 1,
        'AbnormalExtFormActionR': 1,
        'ExtMetaScriptLinkRT': 1,
        'PctExtNullSelfRedirectHyperlinksRT': 1
    }
    if include_extended_features:
        spam_keywords = ['win', 'free', 'urgent', 'lottery', 'click here']
        email_features = {
            'SpamKeywordCount': sum(1 for word in spam_keywords if word.lower() in email_body.lower()),
            'EmailLength': len(email_body),
            'SenderReputation': 0,
            'UrlEntropy': compute_entropy(url)
        }
        features.update(email_features)
    return features

# Attachment feature extraction
def compute_attachment_features(attachment_info):
    try:
        filename, size_str = attachment_info.split(',')
        size = int(size_str.strip().replace('KB', '')) * 1024 if 'KB' in size_str else int(size_str.strip())
        file_ext = filename.split('.')[-1].lower() if '.' in filename else ''
        risky_extensions = ['exe', 'bat', 'js', 'vbs', 'scr']
        features = {
            'FileSize': size,
            'IsRiskyExtension': 1 if file_ext in risky_extensions else 0,
            'FileNameLength': len(filename),
            'FileNameEntropy': compute_entropy(filename)
        }
        return features
    except:
        return {'FileSize': 0, 'IsRiskyExtension': 0, 'FileNameLength': 0, 'FileNameEntropy': 0}

# Prediction functions
def predict_url(url, model, email_body=""):
    features = compute_url_features(url, email_body, include_extended_features=True)
    feature_df = pd.DataFrame([features])
    prediction = model.predict(feature_df)[0]
    return 'Phishing' if prediction == 0 else 'Legitimate'

def predict_anomaly(url, email_body=""):
    features = compute_url_features(url, email_body, include_extended_features=True)
    feature_df = pd.DataFrame([features])
    prediction = anomaly_model.predict(feature_df)[0]
    return 'Anomaly' if prediction == -1 else 'Normal'

def predict_malware(url, email_body=""):
    features = compute_url_features(url, email_body, include_extended_features=True)
    feature_df = pd.DataFrame([features])
    prediction = malware_model.predict(feature_df)[0]
    return 'Malware' if prediction == 1 else 'Safe'

def predict_spam(email_body):
    features = compute_url_features("", email_body, include_extended_features=True)
    feature_df = pd.DataFrame([features])
    prediction = spam_model.predict(feature_df)[0]
    return 'Spam' if prediction == 1 else 'Non-Spam'

def predict_attachment_malware(attachment_info):
    features = compute_attachment_features(attachment_info)
    feature_df = pd.DataFrame([features])
    prediction = attachment_malware_model.predict(feature_df)[0]
    return 'Malware' if prediction == 1 else 'Safe'

# Generate report
def generate_report(url, xgb_result, lgb_result, anomaly_result, malware_result, spam_result, attachment_info, attachment_malware_result):
    return f"""
    <div class="bg-white rounded-xl shadow-lg p-8">
        <h2 class="text-2xl font-semibold text-blue-800 mb-6">Cybersecurity Awareness Report</h2>
        <p class="text-blue-900"><span class="font-semibold">URL:</span> {url}</p>
        <p class="text-blue-900"><span class="font-semibold">XGBoost Classification:</span> {xgb_result}</p>
        <p class="text-blue-900"><span class="font-semibold">LightGBM Classification:</span> {lgb_result}</p>
        <p class="text-blue-900"><span class="font-semibold">Anomaly Detection:</span> {anomaly_result}</p>
        <p class="text-blue-900"><span class="font-semibold">Malware Detection:</span> {malware_result}</p>
        <p class="text-blue-900"><span class="font-semibold">Spam Detection:</span> {spam_result}</p>
        {f'<p class="text-blue-900"><span class="font-semibold">Attachment:</span> {attachment_info}</p><p class="text-blue-900"><span class="font-semibold">Attachment Malware Detection:</span> {attachment_malware_result}</p>' if attachment_info else ''}
        <p class="text-blue-900 mt-4"><span class="font-semibold">Tips for Staff:</span></p>
        <ul class="list-disc pl-5 text-blue-900">
            <li>Verify URLs for suspicious patterns (e.g., excessive dashes, random strings).</li>
            <li>Ensure links use HTTPS; avoid non-secure URLs.</li>
            <li>Do not open unexpected email attachments, especially with risky extensions (e.g., .exe, .js).</li>
            <li>Be cautious of emails with urgent language or suspicious keywords like 'free' or 'win'.</li>
            <li>Report suspicious URLs, attachments, or emails to the IT security team immediately.</li>
        </ul>
    </div>
    """

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        email_body = request.form['email_body']
        attachment_info = request.form.get('attachment_info', '')
        urls = extract_urls(email_body)
        url = urls[0] if urls else ""
        xgb_result = predict_url(url, xgb_model, email_body) if url else "N/A"
        lgb_result = predict_url(url, lgb_model, email_body) if url else "N/A"
        anomaly_result = predict_anomaly(url, email_body) if url else "N/A"
        malware_result = predict_malware(url, email_body) if url else "N/A"
        spam_result = predict_spam(email_body)
        attachment_malware_result = predict_attachment_malware(attachment_info) if attachment_info else "N/A"
        email_snippet = email_body[:100] + '...' if len(email_body) > 100 else email_body
        email_history.append({
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'email_snippet': email_snippet,
            'url': url,
            'result': xgb_result,
            'lgb_result': lgb_result,
            'anomaly_result': anomaly_result,
            'malware_result': malware_result,
            'spam_result': spam_result,
            'attachment_info': attachment_info,
            'attachment_malware_result': attachment_malware_result
        })
        report = generate_report(url, xgb_result, lgb_result, anomaly_result, malware_result, spam_result, attachment_info, attachment_malware_result)
        return render_template_string(HTML_TEMPLATE, url=url, result=xgb_result, lgb_result=lgb_result, anomaly_result=anomaly_result, malware_result=malware_result, spam_result=spam_result, attachment_info=attachment_info, attachment_malware_result=attachment_malware_result, report=report, history=email_history)
    return render_template_string(HTML_TEMPLATE, history=email_history)

if __name__ == '__main__':
    app.run(debug=True)