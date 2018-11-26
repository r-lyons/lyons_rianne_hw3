import edu.stanford.nlp.ie.crf.CRFClassifier
import edu.stanford.nlp.tagger.maxent.{MaxentTagger,TaggerConfig}
import java.io._
import java.util.Properties
import scala.io.Source

object RunBaselines extends App {

    //splitTokens()

    //tagTrainingData()

    //addTrainingLabels()

    //trainCRF()

    runCRF()

    def runCRF() = {

      val crfToTest = CRFClassifier.getClassifier("src/main/resources/crf-model.ser.gz")
      var numICorrect, numOCorrect, numPCorrect = 0
      var totalI, totalP, totalO = 0
      var totalPredI, totalPredP, totalPredO = 0
      val testDocIds = Vector("24330473", "24641638")


      Vector("I", "O", "P"). foreach {testDoc =>
        testDocIds.foreach {id => 
          val testSource = Source.fromFile("src/main/resources/" + testDoc + "/test/" + id + ".tsv")
          val testString = testSource.getLines.mkString
          testSource.close()
          println(testDoc + ":" + id)
          val preds = crfToTest.classifyToString(testString, "slashTags", false)
          println(preds)
          val predScores = calculateScores(preds, id, testDoc)
          testDoc match {
            case "I" => {
              numICorrect += predScores._1
              totalPredI += predScores._2
              totalI += predScores._3
            }
            case "O" => {
              numOCorrect += predScores._1
              totalPredO += predScores._2
              totalO += predScores._3
            }
            case "P" => {
              numPCorrect += predScores._1
              totalPredP += predScores._2
              totalP += predScores._3
            }

          }
        } 
      }
      val precisionI = numICorrect.toDouble / totalPredI.toDouble
      val precisionO = numOCorrect.toDouble / totalPredO.toDouble
      val precisionP = numPCorrect.toDouble / totalPredP.toDouble
      val recallI = numICorrect.toDouble / totalI.toDouble
      val recallO = numOCorrect.toDouble / totalO.toDouble
      val recallP = numPCorrect.toDouble / totalP.toDouble
      val fScoreI = 2 * ((precisionI*recallI) / (precisionI+recallI))
      val fScoreO = 2 * ((precisionO*recallO) / (precisionO+recallO))
      val fScoreP = 2 * ((precisionP*recallP) / (precisionP+recallP))

      println("I Precision: " + precisionI.toString)
      println("O Precision: " + precisionO.toString)
      println("P Precision: " + precisionP.toString)
      println("I Recall: " + recallI.toString)
      println("O Recall: " + recallO.toString)
      println("P Recall: " + recallP.toString)
      println("I F-Score: " + fScoreI.toString)
      println("O F-Score: " + fScoreO.toString)
      println("P F-Score: " + fScoreP.toString)


    }

    def calculateScores(wordsAndLabels: String, docId: String, docLabel: String) : (Int, Int, Int) = {
      
      val wPV = wordsAndLabels.split(" ").toVector
      val docSource = Source.fromFile("src/main/resources/" + docLabel + "/test/" + docId + "_AGGREGATED.ann")
      val docGold = docSource.getLines.mkString.split(",").toVector
      docSource.close()
      var numCorrectPreds = 0
      var totalPreds = 0
      val wordPredVec = wPV.take(docGold.length) // for length mismatch error
      wordPredVec.foreach {wordPred =>
        val goldLabel = docGold(wordPredVec.indexOf(wordPred))
        wordPred.takeRight(1) match {
          case gl if goldLabel == wordPred.takeRight(1)  => {
            numCorrectPreds += 1
            totalPreds += 1
            }
          case "/" =>
          case _ => totalPreds += 1
        }
      }
      return (numCorrectPreds, totalPreds, docGold.length)

    }

    def trainCRF() = {

      val pFile = new File("src/main/scala/trainProperties.prop")
      val pBW = new BufferedReader(new FileReader(pFile))
      val props = new Properties()
      props.load(pBW)
      pBW.close()

      val crfToTrain = new CRFClassifier(props)
      crfToTrain.train()  
      crfToTrain.serializeClassifier(props.getProperty("serializeTo"))

    }


    def addTrainingLabels() = {

      val trainDocIds = Vector("125133", "161652", "219802", "43164", "88754")


      Vector("I", "O", "P").foreach {trainDoc =>
        trainDocIds.foreach {id => 

          val wordTagSource = Source.fromFile("/home/rianne/Documents/CSC585/assignment3/" + trainDoc + "/train/" + id + ".tsv")
          val wordTag = wordTagSource.getLines.toVector
          val labelSource = Source.fromFile("/home/rianne/Documents/CSC585/assignment3/" + trainDoc + "/train/" + id + "_AGGREGATED.ann")
          val labelVec = labelSource.getLines.mkString.split(",").toVector
          val wordTagLabel = wordTag.map(w => w.mkString + "\t" + labelVec(wordTag.indexOf(w)).toString + "\n")
          wordTagSource.close()
          labelSource.close()
      
          val wtlFile = new File("/home/rianne/Documents/CSC585/assignment3/" + trainDoc + "/train/" + id + ".tsv")
          val wtlBW = new BufferedWriter(new FileWriter(wtlFile))
          wtlBW.write(wordTagLabel.mkString)
          wtlBW.close()
        }
      }

    }


    def tagTrainingData() = {

      val tagProperties = new TaggerConfig("-model", "edu/stanford/nlp/models/pos-tagger/wsj-0-18-left3words-nodistsim.tagger", "-tokenize", "false", "-tagSeparator", "\t", "-outputFile", "/home/rianne/Documents/CSC585/assignment3/I/train/125133.tsv", "-outputFormat", "tsv"/*"-textFile", "/home/rianne/Documents/CSC585/assignment3/I/train/125133.tokens"*/)

      val tagger = new MaxentTagger(tagProperties)
      val trainDocIds = Vector("125133", "161652", "219802", "43164", "88754")

      
      Vector("I", "O", "P").foreach {trainDoc =>
        trainDocIds.foreach {id => 
          val textSource = Source.fromFile("/home/rianne/Documents/CSC585/assignment3/" + trainDoc + "/train/" + id + ".tokens")
          val textToTag = textSource.getLines.mkString
          val docTags = tagger.tagTokenizedString(textToTag).mkString.replaceAll(" ", "\n")
          textSource.close()

          val tgFile = new File("/home/rianne/Documents/CSC585/assignment3/" + trainDoc + "/train/" + id + ".tsv")
          val tgBW = new BufferedWriter(new FileWriter(tgFile))
          tgBW.write(docTags)
          tgBW.close()
        }
      }


    }

    def splitTokens() = {
      val trainDocIds = Vector("125133", "161652", "219802", "43164", "88754")
      val testDocIds = Vector("24330473", "24641638")

      Vector("I", "O", "P").foreach {trainLabel => 
        trainDocIds.foreach {id => 

          //modify intervention tokens to get one sentence per line
          val docTrainSource = Source.fromFile("/home/rianne/Documents/CSC585/assignment3/" + trainLabel + "/train/" + id + ".tokens")
          val newTrainTokenString = docTrainSource.getLines.mkString.replaceAll("""\.\s""", "\\.\n")
          docTrainSource.close()
          val docTrainPW = new PrintWriter("/home/rianne/Documents/CSC585/assignment3/" + trainLabel + "/train/" + id + ".tokens")
          docTrainPW.write(newTrainTokenString)
          docTrainPW.close
          }
        
      }
      

      //split testing data, save as .tsv
      Vector("I", "O", "P").foreach {testLabel =>
        testDocIds.foreach {id =>
          val docTestSource = Source.fromFile("src/main/resources/" + testLabel + "/test/" + id + ".tokens")
          val newTestTokenString = docTestSource.getLines.mkString.replaceAll("""\.\s""", "\\.\n")
          docTestSource.close()
          val docTestPW = new PrintWriter("src/main/resources/" + testLabel + "/test/" + id + ".tsv")
          docTestPW.write(newTestTokenString)
          docTestPW.close
          }
        }
    }
    



}
