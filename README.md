# Software Price Prediction

A machine learning project to predict software prices based on various features.

## Project Structure

- `src/price.py` - Main training and prediction script
- `data/software_prices_large.csv` - Dataset 

  
## 🚀 Features

- **Data Preprocessing**: Handles categorical variables and feature engineering
- **Machine Learning**: Gradient Boosting Regressor for accurate predictions
- **User Interaction**: Command-line interface for real-time price predictions
- **Model Evaluation**: Comprehensive metrics including R² score
- **Scalable Architecture**: Modular code structure for easy maintenance

## 📊 Dataset Features

The model uses the following features for prediction:
- **Category**: Software type (Productivity, Security, Development, etc.)
- **Platform**: Operating system compatibility (Windows, Mac, Linux, etc.)
- **Subscription Type**: License model (Monthly, Annual, One-time, etc.)
- **Features**: Number of features offered
- **Users Supported**: Maximum number of concurrent users
- **Release Year**: Year of software release
- **Years Since Release**: Engineered feature (calculated from release year)

## 🛠️ Installation
 
### Prerequisites
- Python 3.8 or higher
- pip (Python package manager)

### Steps

1. **Clone the repository**:
```bash
git clone https://github.com/yourusername/software-price-prediction.git
cd software-price-prediction

