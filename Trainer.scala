package com.sparkProject

import org.apache.spark.SparkConf
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature._
import org.apache.spark.ml.tuning.{ParamGridBuilder, TrainValidationSplit, TrainValidationSplitModel}
import org.apache.spark.sql.{DataFrame, SparkSession}


object Trainer {

  def main(args: Array[String]): Unit = {

    val conf = new SparkConf().setAll(Map(
      "spark.scheduler.mode" -> "FIFO",
      "spark.speculation" -> "false",
      "spark.reducer.maxSizeInFlight" -> "48m",
      "spark.serializer" -> "org.apache.spark.serializer.KryoSerializer",
      "spark.kryoserializer.buffer.max" -> "1g",
      "spark.shuffle.file.buffer" -> "32k",
      "spark.default.parallelism" -> "12",
      "spark.sql.shuffle.partitions" -> "12",
      "spark.driver.maxResultSize" -> "2g"
    ))

    val spark = SparkSession
      .builder
      .config(conf)
      .appName("TP_spark")
      .getOrCreate()


    /** *****************************************************************************
      *
      * fichier Jerome Divac
      *
      *
      * *******************************************************************************/








    /** *****************************************************************************
      *
      * TP 4-5
      *
      *       - lire le fichier sauvegarder précédemment
      *       - construire les Stages du pipeline, puis les assembler
      *       - trouver les meilleurs hyperparamètres pour l'entraînement du pipeline avec une grid-search
      *       - Sauvegarder le pipeline entraîné
      *
      * if problems with unimported modules => sbt plugins update
      *
      * *******************************************************************************/



    /** On commence par charger le Dataframe **/

    val df: DataFrame = spark
      .read
      .option("header", true) // Use first line of all files as header
      .option("inferSchema", "true") // Try to infer the data types of each column (ne sert pas à grand chose ici, car il met en string et retraiter au e))
      .option("nullValue", "false") // replace strings "false" (that indicates missing data) by null values
      .parquet("/home/jerome/Bureau/TP_ParisTech_2017_2018_starter/TP_ParisTech_2017_2018_starter/data/prepared_trainingset")


    /** ici in affiche le nombre de lignes et de colonnes **/

    println(s"Total number of rows: ${df.count}")
    println(s"Number of columns ${df.columns.length}")

    //df.printSchema()


    /** Utilisation de l'algorithme TF-IDF **/

    /** 1er Stage. On sépare les textes en mots (tokens) avec un tokenizer **/

    val tokenizer = new RegexTokenizer()
      .setPattern("\\W+")
      .setGaps(true)
      .setInputCol("text")
      .setOutputCol("tokens")


    /** 2eme Stage. On retire les stop words qui ne contiennent pas beaucoup d'information
        comme (le, la, à...). On utilise la classe StopWordsRemover **/

    val remover = new StopWordsRemover()
      .setInputCol("tokens")
      .setOutputCol("text_filtered")


    /** 3eme Stage. La partie TF de TF-IDF est faite avec la classe CountVectorizer **/

    val vectorizer = new CountVectorizer()
      .setInputCol("text_filtered")
      .setOutputCol("vectorized")


    /** 4eme Stage. Avec IDF, on écrit l’output de cette étape dans une colonne nommée “tfidf” **/

    val idf = new IDF()
      .setInputCol("vectorized")
      .setOutputCol("tfidf")



    /** 5eme Stage. On convertit la variable catégorielle “country2” en données numériques
        Et on renseigne le résultat dans une colonne "country_indexed" **/

    val  index_country = new StringIndexer()
      .setInputCol("country2")
      .setOutputCol("country_indexed")


    /**6eme Stage. On convertit la variable catégorielle “currency2” en données numériques
       Et on renseigne le résultat dans une colonne "currency_indexed" **/

    val index_currency = new StringIndexer()
      .setInputCol("currency2")
      .setOutputCol("currency_indexed")



    /** 7eme Stage . On assemble les features "tfidf", "days_campaign","hours_prepa", "goal",
        "country_indexed" et "currency_indexed"  dans une seule colonne nommée “features”
        On utilise la classe VectorAssembler pour créer ces vecteurs **/

    val assembler = new VectorAssembler()
      .setInputCols(Array("tfidf", "days_campaign","hours_prepa", "goal", "country_indexed", "currency_indexed"))
      .setOutputCol("features")




    /**8eme Stage. Le modèle de classification retenu est une régression logistique
       définie avec les paramètres ci-dessous **/

    val lr = new LogisticRegression()
      .setElasticNetParam(0.0)
      .setFitIntercept(true)
      .setFeaturesCol("features")
      .setLabelCol("final_status")
      .setStandardization(true)
      .setPredictionCol("predictions")
      .setRawPredictionCol("raw_predictions")
      .setThresholds(Array(0.7, 0.3))
      .setTol(1.0e-6)
      .setMaxIter(300)



    /** PIPELINE **/


    /**On va créer le pipeline en assemblant les 8 stages définis précédemment, dans le bon ordre
       Il s'agit d'une méthode pour enchainer successivement différents types de traitements et
       et transformations sur notre jeu de données **/

    val pipeline = new Pipeline()
      .setStages(Array(tokenizer,remover,vectorizer,idf,index_country,index_currency,assembler,lr))




    /** TRAINING AND GRID-SEARCH **/

    /**On va créer un dataFrame nommé “training” et un autre nommé “test” à partir du dataFrame
    de départ de façon à le séparer en training et test sets dans les proportions
    90%, 10% respectivement **/



    val Array(training, test) = df.randomSplit(Array[Double](0.9, 0.1))


    /** on crée maintenant une grille de paramètres pour la grid-search avec différentes valeurs
      * pour la régularisation, le paramètre minDF de countVectorizer **/


    val param_grid = new ParamGridBuilder()
      .addGrid(lr.regParam, Array[Double](10e-8, 10e-6, 10e-4, 10e-2))
      .addGrid(vectorizer.minDF,  Array[Double](55, 75, 95))
      .build()


    /** on utilise MulticlassClassificationEvaluator avec 2 inputs: final_status et predictions **/


    val evaluator = new MulticlassClassificationEvaluator()
      .setLabelCol("final_status")
      .setPredictionCol("predictions")


    /** on utilise ici un ratio de 70% pour l'entrainement **/

    val trainValidationSplit = new TrainValidationSplit()
      .setEstimator(pipeline)
      .setEvaluator(evaluator)
      .setEstimatorParamMaps(param_grid)
      .setTrainRatio(0.7)


    //print("mémoire OK jusqu'ici")

    /** on sélectionne maintenant le meilleur modèle obtenu **/

    val model = trainValidationSplit.fit(training)

    val df_with_predictions = model.transform(test)

    /** et on va afficher le F1-score **/

    println("f1_score = " + evaluator.setMetricName("f1").evaluate(df_with_predictions))

    df_with_predictions.groupBy("final_status", "predictions").count.show()

    model.write.overwrite().save("TP_SPARK_Log_regression_model")


    //println(model.bestModel.)
    //println(model.extractParamMap())



  }


}