package org.example.machinelearning;

import weka.classifiers.trees.RandomForest;
import weka.core.*;
import weka.core.converters.ArffLoader;
import weka.classifiers.Evaluation;

import org.jfree.chart.ChartFactory;
import org.jfree.chart.ChartPanel;
import org.jfree.chart.JFreeChart;
import org.jfree.data.general.DefaultPieDataset;

import javax.swing.*;
import java.io.File;
import java.util.HashMap;
import java.util.Map;

public class RandomForestClassifier {

    public static void main(String[] args) throws Exception {
        // Charger le fichier ARFF
        ArffLoader loader = new ArffLoader();
        loader.setFile(new File("C:/Users/Hp/IdeaProjects/s_test/table.arff"));
        Instances data = loader.getDataSet();

        // Définir l'attribut cible (classe) comme "hardskills"
        data.setClassIndex(data.attribute("niveau_etudes").index());

        // Diviser les données en 60% pour l'entraînement et 40% pour le test
        int trainSize = (int) Math.round(data.numInstances() * 0.6);
        int testSize = data.numInstances() - trainSize;
        Instances trainData = new Instances(data, 0, trainSize);
        Instances testData = new Instances(data, trainSize, testSize);

        // Créer le modèle Random Forest
        RandomForest rf = new RandomForest();
        rf.buildClassifier(trainData); // Entraîner le modèle sur les données d'entraînement

        // Évaluer le modèle sur les données de test
        Evaluation evaluation = new Evaluation(trainData);
        evaluation.evaluateModel(rf, testData);

        // Afficher les résultats de l'évaluation
        System.out.println("Evaluation:");
        // Formater l'accuracy à 2 décimales
        System.out.println("Accuracy: " + String.format("%.2f", evaluation.pctCorrect()) + "%");
        System.out.println("Confusion Matrix: " + evaluation.toMatrixString());
        System.out.println("Detailed Accuracy: " + evaluation.toSummaryString());

        // Créer un dataset pour le graphique circulaire
        Map<String, Integer> predictionCounts = new HashMap<>();

        // Prédire pour chaque instance du jeu de test
        for (int i = 0; i < testData.numInstances(); i++) {
            Instance instance = testData.instance(i);
            double predictedClass = rf.classifyInstance(instance);

            // Récupérer la compétence technique prédite
            String predictedSkills = instance.classAttribute().value((int) predictedClass);

            // Compter les prédictions pour chaque compétence
            predictionCounts.put(predictedSkills, predictionCounts.getOrDefault(predictedSkills, 0) + 1);
        }

        // Créer un dataset pour le graphique circulaire
        DefaultPieDataset dataset = new DefaultPieDataset();
        for (Map.Entry<String, Integer> entry : predictionCounts.entrySet()) {
            dataset.setValue(entry.getKey(), entry.getValue());
        }

        // Créer le graphique circulaire
        JFreeChart pieChart = ChartFactory.createPieChart(
                "Répartition du niveaux d'etudes prédites",  // Titre
                dataset,  // Dataset
                true,  // Légende
                true,  // Tooltip
                false  // URL
        );

        // Afficher le graphique circulaire dans une fenêtre
        JPanel chartPanel = new ChartPanel(pieChart);
        chartPanel.setPreferredSize(new java.awt.Dimension(800, 600));
        JFrame frame = new JFrame();
        frame.setDefaultCloseOperation(JFrame.DISPOSE_ON_CLOSE);
        frame.getContentPane().add(chartPanel);
        frame.pack();
        frame.setVisible(true);
    }
}
