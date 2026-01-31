package org.example.machinelearning;
import weka.core.*;
import weka.classifiers.functions.Logistic;
import weka.classifiers.Evaluation;
import weka.core.converters.ConverterUtils.DataSource;
import org.jfree.chart.ChartFactory;
import org.jfree.chart.ChartPanel;
import org.jfree.chart.JFreeChart;
import org.jfree.data.general.DefaultPieDataset;
import javax.swing.*;
import java.util.Random;

public class LogisticRegressionMultiClass {

    public static void main(String[] args) {
        try {
            // Charger le fichier ARFF
            DataSource source = new DataSource("C:/Users/Hp/IdeaProjects/s_test/table.arff");
            Instances data = source.getDataSet();

            // Vérifier si les données sont chargées correctement
            System.out.println("Nombre d'instances chargées : " + data.numInstances());

            // Vérifier l'attribut cible
            data.setClassIndex(1);  // ou utilisez le bon index si "contrat_type" n'est pas la dernière colonne
            System.out.println("Attribut cible : " + data.classAttribute().name());

            // Mélanger les données pour une division aléatoire
            data.randomize(new Random(42));  // Utilisation d'une graine pour la reproductibilité

            // Diviser les données en 60% pour l'entraînement et 40% pour le test
            int trainSize = (int) Math.round(data.numInstances() * 0.7);
            int testSize = data.numInstances() - trainSize;
            Instances trainData = new Instances(data, 0, trainSize);
            Instances testData = new Instances(data, trainSize, testSize);

            // Créer un classifieur de régression logistique multiclasse
            Logistic model = new Logistic();

            // Entraîner le modèle sur les données d'entraînement
            model.buildClassifier(trainData);

            // Évaluation du modèle avec les données de test
            Evaluation eval = new Evaluation(trainData);
            eval.evaluateModel(model, testData);

            // Afficher les résultats de l'évaluation
            System.out.println("Résultats de l'évaluation :");
            System.out.println(eval.toSummaryString());
            System.out.println("Matrice de confusion : ");
            System.out.println(eval.toMatrixString());

            // Afficher l'accuracy avec 2 chiffres après la virgule
            double accuracy = eval.pctCorrect();
            System.out.println("Précision : " + String.format("%.2f", accuracy) + "%");

            // Créer un tableau pour compter les prédictions pour chaque classe
            int[] predictedCounts = new int[data.numClasses()];

            // Prédire les classes pour chaque instance de test
            for (int i = 0; i < testData.numInstances(); i++) {
                double predictedClass = model.classifyInstance(testData.instance(i));  // Prédiction de la classe
                predictedCounts[(int) predictedClass]++;  // Incrémenter le compteur pour la classe prédite
            }

            // Créer un dataset pour le graphique circulaire des prédictions
            DefaultPieDataset predictionDataset = new DefaultPieDataset();
            for (int i = 0; i < data.numClasses(); i++) {
                predictionDataset.setValue(data.classAttribute().value(i), predictedCounts[i]);
            }

            // Créer le graphique circulaire pour les prédictions
            JFreeChart predictionChart = ChartFactory.createPieChart(
                    "Répartition des Prédictions",       // Titre du graphique
                    predictionDataset,                   // Données
                    true,                                // Légende
                    true,                                // Informations sur les sections
                    false                                // URL
            );

            // Afficher le graphique
            ChartPanel predictionPanel = new ChartPanel(predictionChart);
            predictionPanel.setPreferredSize(new java.awt.Dimension(800, 600));
            JFrame predictionFrame = new JFrame("Graphique Circulaire des Prédictions");
            predictionFrame.setDefaultCloseOperation(JFrame.DISPOSE_ON_CLOSE);
            predictionFrame.getContentPane().add(predictionPanel);
            predictionFrame.pack();
            predictionFrame.setVisible(true);

        } catch (Exception e) {
            System.out.println("Erreur : " + e.getMessage());
            e.printStackTrace();
        }
    }
}
