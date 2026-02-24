# 📈 Django Trading App (Showcase Project)

This repository contains a Django-based trading analytics web application built as a portfolio project to demonstrate:

- Django backend development
- Data ingestion from CSV sources
- Modular configuration via environment variables
- Market / ticker extensibility
- Clean project structure and reproducibility

> ⚠️ This is a showcase project intended for demonstration only.  
> It is not intended for production trading or financial advice.

---

## 🚀 Project Overview

The application allows users to:

- Load market data from CSV files
- Explore trading signals and analytics
- Visualize trading-related information via a Django-powered web interface
- Dynamically switch data sources via environment configuration

The architecture is intentionally simple and transparent to highlight:

- Clean separation between data source and application logic
- Django project structure
- Config-driven execution

---

## 🧠 Skills Demonstrated

- Django project setup and configuration
- Environment-driven application behavior
- Data-driven application architecture
- Backend structuring for analytics workflows
- Modular directory-based data management
- Reproducible local development setup

---

## 🛠️ Installation

Clone the repository:

    git clone https://github.com/monti-nicolas/django-trading-app.git
    cd django-trading-app

Create and activate a virtual environment:

    python -m venv venv
    source venv/bin/activate  # Mac/Linux
    venv\Scripts\activate     # Windows

Install dependencies:

    pip install -r requirements.txt

---

## ▶️ Usage (Demo Mode)

For demonstration purposes, you can run the application using the provided demo dataset.

Set the data source directory:

    export SOURCE_DIR=source/demo

Then start the Django development server:

    python manage.py runserver 8000

Open your browser and navigate to:

    http://127.0.0.1:8000/

---

## 📂 Data Source Configuration

The application dynamically reads CSV files from the directory specified by the SOURCE_DIR environment variable.

### Demo Data

    source/demo

This directory contains example CSV files used for demonstration.

### Extending to Other Markets or Tickers

You can replicate the source/demo directory structure to support:

- Different markets
- Different tickers
- Alternative datasets

For example:

    source/sp500
    source/asx
    source/crypto

Then update the environment variable:

    export SOURCE_DIR=source/sp500

⚠️ Important:  
All CSV files must have exactly the same file names and schema as those in source/demo, since the application expects a consistent structure.

---

## 🏗️ Project Structure (High-Level)

    django-trading-app/
    │
    ├── source/
    │   ├── demo/
    │   └── ...
    │
    ├── trading_app/
    ├── manage.py
    └── requirements.txt

- source/ → Market data inputs (CSV-based)
- Django app → Application logic and views
- manage.py → Django management entry point

---

## ⚠️ Disclaimer

This project:

- Is for educational and demonstration purposes only
- Does not execute real trades
- Does not connect to brokerage APIs
- Should not be used for financial decision-making

---

## 📌 Why This Project Exists

This repository is designed to showcase:

- Backend engineering ability
- Data-centric application design
- Django fluency
- Clean configuration patterns
- Ability to structure analytical web applications
