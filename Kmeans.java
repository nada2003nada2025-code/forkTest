package org.example.machinelearning;
import org.jfree.chart.ChartFactory;
import org.jfree.chart.ChartPanel;
import org.jfree.chart.JFreeChart;
import org.jfree.data.category.DefaultCategoryDataset;

import weka.clusterers.SimpleKMeans;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

import javax.swing.*;
import java.util.HashMap;
import java.util.Map;

public class Kmeans {

    public static void main(String[] args) {
        try {
            // Charger le fichier ARFF
            String filePath = "C:/Users/Hp/IdeaProjects/s_test/table.arff"; // Remplace par le chemin de ton fichier
            DataSource source = new DataSource(filePath);
            Instances data = source.getDataSet();

            // Vérifier si l'attribut "post" existe
            if (data.attribute("post") == null) {
                throw new IllegalArgumentException("L'attribut 'post' est introuvable dans le fichier ARFF.");
            }

            // Diviser les données : 60% entraînement, 40% test
            int trainSize = (int) Math.round(data.numInstances() * 0.8);
            int testSize = data.numInstances() - trainSize;

            Instances trainingData = new Instances(data, 0, trainSize);

            // Méthode du coude pour déterminer le bon nombre de clusters (k)
            int optimalK = findOptimalK(trainingData);

            // Appliquer K-Means avec le nombre de clusters optimal
            SimpleKMeans kmeans = new SimpleKMeans();
            kmeans.setNumClusters(optimalK);
            kmeans.buildClusterer(trainingData);

            // Identifier l'attribut "post" (emploi)
            int postIndex = data.attribute("post").index();

            // Initialiser les comptages par cluster pour l'entraînement
            Map<Integer, Map<String, Integer>> clusterJobCounts = new HashMap<>();

            // Parcourir les instances d'entraînement et assigner aux clusters
            for (int i = 0; i < trainingData.numInstances(); i++) {
                Instance instance = trainingData.instance(i);
                int cluster = kmeans.clusterInstance(instance); // Assigner le cluster
                String job = instance.stringValue(postIndex); // Obtenir la valeur de "post"

                // Mettre à jour les comptages
                clusterJobCounts.putIfAbsent(cluster, new HashMap<>());
                Map<String, Integer> jobCounts = clusterJobCounts.get(cluster);
                jobCounts.put(job, jobCounts.getOrDefault(job, 0) + 1);
            }

            // Vérifier que tous les clusters ont été traités
            for (int i = 0; i < optimalK; i++) {
                if (!clusterJobCounts.containsKey(i)) {
                    System.out.println("Cluster " + i + " est vide ou n'a pas d'emplois.");
                    clusterJobCounts.put(i, new HashMap<>());
                }
            }

            // Afficher les emplois les plus demandés par cluster
            System.out.println("Emplois les plus demandés par cluster dans l'ensemble d'entraînement :");
            Map<String, Integer> mostDemandedJobs = new HashMap<>();

            for (Map.Entry<Integer, Map<String, Integer>> entry : clusterJobCounts.entrySet()) {
                int cluster = entry.getKey();
                Map<String, Integer> jobCounts = entry.getValue();

                // Trouver l'emploi le plus fréquent
                String mostDemandedJob = null;
                int maxCount = 0;
                for (Map.Entry<String, Integer> jobEntry : jobCounts.entrySet()) {
                    if (jobEntry.getValue() > maxCount) {
                        mostDemandedJob = jobEntry.getKey();
                        maxCount = jobEntry.getValue();
                    }
                }

                System.out.println("Cluster " + cluster + " : " + mostDemandedJob + " (" + maxCount + " occurrences)");
                mostDemandedJobs.put(mostDemandedJob, maxCount);
            }

            // Afficher les centroids des clusters
            Instances centroids = kmeans.getClusterCentroids();
            System.out.println("Centroides des clusters :");
            for (int i = 0; i < centroids.numInstances(); i++) {
                Instance centroid = centroids.instance(i);
                System.out.println("Cluster " + i + " centroid: " + centroid);
            }

            // Création du graphique à barres
            DefaultCategoryDataset dataset = new DefaultCategoryDataset();
            for (Map.Entry<String, Integer> entry : mostDemandedJobs.entrySet()) {
                dataset.addValue(entry.getValue(), "Emploi", entry.getKey());
            }

            // Créer le graphique
            JFreeChart barChart = ChartFactory.createBarChart(
                    "Emplois les plus demandés par cluster", // Titre du graphique
                    "Emploi", // Axe des X
                    "Occurrences", // Axe des Y
                    dataset // Données
            );

// Réduire la largeur des barres
            org.jfree.chart.plot.CategoryPlot plot = barChart.getCategoryPlot();
            org.jfree.chart.renderer.category.BarRenderer renderer = (org.jfree.chart.renderer.category.BarRenderer) plot.getRenderer();
            renderer.setMaximumBarWidth(0.05); // Ajuste cette valeur (0.05) selon tes préférences

// Afficher le graphique dans un panel
            ChartPanel chartPanel = new ChartPanel(barChart);
            chartPanel.setPreferredSize(new java.awt.Dimension(800, 600));
            JFrame frame = new JFrame();
            frame.setDefaultCloseOperation(JFrame.DISPOSE_ON_CLOSE); // Fermer uniquement cette fenêtre
            frame.getContentPane().add(chartPanel);
            frame.pack();
            frame.setVisible(true);


        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    // Méthode pour calculer la méthode du coude et trouver k optimal
    public static int findOptimalK(Instances trainingData) {
        double[] sseValues = new double[10]; // Tester de k = 1 à k = 10

        for (int k = 1; k <= 10; k++) {
            try {
                SimpleKMeans kmeans = new SimpleKMeans();
                kmeans.setNumClusters(k);
                kmeans.buildClusterer(trainingData);

                // Calculer la somme des erreurs quadratiques (SSE)
                double sumSquaredErrors = 0.0;
                for (int i = 0; i < trainingData.numInstances(); i++) {
                    Instance instance = trainingData.instance(i);
                    int cluster = kmeans.clusterInstance(instance);
                    double distance = kmeans.getDistanceFunction().distance(instance, kmeans.getClusterCentroids().instance(cluster));
                    sumSquaredErrors += Math.pow(distance, 2);
                }
                sseValues[k - 1] = sumSquaredErrors;
                System.out.println("k = " + k + " SSE = " + sumSquaredErrors);
            } catch (Exception e) {
                e.printStackTrace();
            }
        }

        // Identifier le "coude"
        int optimalK = 1;
        double minDifference = Double.MAX_VALUE;
        for (int k = 1; k < sseValues.length - 1; k++) {
            double diff = sseValues[k] - sseValues[k - 1];
            if (diff < minDifference) {
                minDifference = diff;
                optimalK = k + 1; // 1-based indexing
            }
        }

        System.out.println("Le nombre optimal de clusters est : " + optimalK);
        return optimalK;
    }
}
