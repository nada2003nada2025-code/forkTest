package org.example.machinelearning;

import javax.swing.*;
import java.awt.*;
import weka.classifiers.trees.J48;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.classifiers.Evaluation;
import org.jfree.chart.ChartFactory;
import org.jfree.chart.ChartPanel;
import org.jfree.chart.JFreeChart;
import org.jfree.data.general.DefaultPieDataset;
import java.util.Random;

public class ClassificationTask {
    public static void main(String[] args) {
        SwingUtilities.invokeLater(() -> {
            try {
                // Charger le fichier ARFF
                String arffFile = "C:/Users/Hp/IdeaProjects/s_test/table.arff";
                DataSource source = new DataSource(arffFile);
                Instances data = source.getDataSet();
                data.setClassIndex(data.numAttributes() - 1);

                // Mélanger les données pour une division aléatoire
                data.randomize(new Random(1));

                // Diviser les données en 60% pour l'entraînement et 40% pour les tests
                int trainSize = (int) (data.numInstances() * 0.7);
                int testSize = data.numInstances() - trainSize;
                Instances trainData = new Instances(data, 0, trainSize);
                Instances testData = new Instances(data, trainSize, testSize);

                // Créer et entraîner le modèle J48
                J48 tree = new J48();
                tree.buildClassifier(trainData);

                // Effectuer une évaluation sur les données de test
                Evaluation eval = new Evaluation(trainData);
                eval.evaluateModel(tree, testData);

                // Afficher les résultats dans la console
                System.out.println("Accuracy: " + String.format("%.2f", eval.pctCorrect()) + "%");
                System.out.println("Confusion Matrix:\n" + eval.toMatrixString());
                System.out.println("Precision & Recall:\n" + eval.toSummaryString());
                System.out.println("Example Predictions:\n" + getExamplePredictions(testData, tree));
                System.out.println("Sector Distribution (Precision per Class):\n" + getSectorDistribution(eval, testData));

                // Créer l'interface graphique
                JFrame frame = new JFrame("Model Evaluation Results");
                frame.setDefaultCloseOperation(JFrame.DISPOSE_ON_CLOSE); // Fermer uniquement cette fenêtre
                frame.setSize(700, 600);
                frame.setLayout(new BorderLayout());

                // Panel principal pour le graphique circulaire
                JPanel panel = new JPanel();
                panel.setLayout(new BoxLayout(panel, BoxLayout.Y_AXIS));

                // Ajouter le graphique circulaire pour la distribution des secteurs
                panel.add(createPieChartPanel(testData));

                // Ajouter le panel au frame
                frame.add(panel, BorderLayout.CENTER);

                // Afficher la fenêtre
                frame.setVisible(true);

            } catch (Exception e) {
                e.printStackTrace();
            }
        });
    }

    // Méthode pour obtenir des exemples de prédictions
    private static String getExamplePredictions(Instances data, J48 tree) throws Exception {
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < 5; i++) {
            double predictedClass = tree.classifyInstance(data.instance(i));
            sb.append("Example ").append(i + 1).append(": ");
            sb.append("Predicted: ").append(data.classAttribute().value((int) predictedClass));
            sb.append(" | Actual: ").append(data.instance(i).stringValue(data.classIndex()));
            sb.append("\n");
        }
        return sb.toString();
    }

    // Méthode pour obtenir la distribution des secteurs (précision par classe)
    private static String getSectorDistribution(Evaluation eval, Instances data) {
        StringBuilder sb = new StringBuilder();
        int numClasses = data.classAttribute().numValues();

        // Calcul de la précision par classe
        for (int i = 0; i < numClasses; i++) {
            double precision = eval.precision(i);
            sb.append(data.classAttribute().value(i)).append(": ");
            sb.append(String.format("%.2f", precision * 100)).append("%\n");
        }
        return sb.toString();
    }

    // Méthode pour créer le graphique circulaire montrant la distribution des secteurs
    private static JPanel createPieChartPanel(Instances data) {
        DefaultPieDataset dataset = new DefaultPieDataset();

        // Compter la fréquence des secteurs dans les données
        for (int i = 0; i < data.numInstances(); i++) {
            String sector = data.instance(i).stringValue(data.classIndex());

            // Vérifier si la clé existe déjà dans le dataset, sinon l'ajouter
            if (dataset.getKeys().contains(sector)) {
                // Incrémenter la valeur existante
                double currentValue = dataset.getValue(sector).doubleValue();
                dataset.setValue(sector, currentValue + 1);
            } else {
                // Ajouter une nouvelle clé avec la valeur 1
                dataset.setValue(sector, 1);
            }
        }

        // Créer un graphique circulaire à partir des données
        JFreeChart chart = ChartFactory.createPieChart(
                "Sector Distribution", // titre du graphique
                dataset,               // jeu de données
                true,                  // afficher la légende
                true,                  // outil d'info
                false                  // URL
        );

        // Créer un panel pour afficher le graphique
        ChartPanel chartPanel = new ChartPanel(chart);
        chartPanel.setPreferredSize(new java.awt.Dimension(400, 300));
        return chartPanel;
    }
}
