import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from monitoring.performance_tracker import PerformanceTracker
from monitoring.drift_monitor import DriftMonitor
import os
import glob
from datetime import datetime

def get_latest_log_file(log_dir='logs', prefix='performance_logs_'):
    pattern = os.path.join(log_dir, f'{prefix}*.csv')
    files = glob.glob(pattern)
    if not files:
        return None
    return max(files, key=os.path.getctime)

def main():
    st.title("Dashboard de Surveillance des Modèles")

    performance_tracker = PerformanceTracker()
    drift_monitor = DriftMonitor()

    # Surveillance de la Performance
    st.header("Surveillance de la Performance")
    
    current_log_file = get_latest_log_file()
    if current_log_file:
        st.text(f"Fichier de log en cours : {current_log_file}")
        
        try:
            df = pd.read_csv(current_log_file)
            st.text(f"Nombre d'entrées dans le fichier de log : {len(df)}")
            st.text(f"Colonnes dans le fichier de log : {df.columns}")
            st.text(f"Aperçu des données :\n{df.head().to_string()}")
            
            overall_accuracy, class_accuracies = performance_tracker.get_performance_metrics(log_file=current_log_file)
            
            if overall_accuracy is not None:
                st.metric("Précision Globale", f"{overall_accuracy:.2%}")
            else:
                st.warning("Données insuffisantes pour calculer la précision globale")

            # Graphique de précision par classe
            st.subheader("Précision par Classe")
            
            valid_class_accuracies = {k: v for k, v in class_accuracies.items() if v is not None}
            
            if valid_class_accuracies:
                fig, ax = plt.subplots()
                ax.bar(valid_class_accuracies.keys(), valid_class_accuracies.values())
                plt.xticks(rotation=90)
                st.pyplot(fig)
            else:
                st.warning("Pas de données valides pour afficher la précision par classe")

            # Surveillance de la Dérive
            st.header("Surveillance de la Dérive")
            drift_detected, drift_details = drift_monitor.check_drift()
            if drift_detected:
                st.warning(f"Dérive détectée: {drift_details}")
            else:
                st.success("Aucune dérive détectée")

            # Affichage des prédictions récentes
            st.header("Prédictions Récentes")
            if not df.empty:
                st.dataframe(df.tail())
            else:
                st.info("Aucune prédiction récente à afficher")

        except Exception as e:
            st.error(f"Erreur lors de la lecture du fichier de log : {str(e)}")
    else:
        st.warning("Aucun fichier de log trouvé.")

if __name__ == "__main__":
    main()