name := "lyons_rianne_hw3"

version := "1.0.2"

scalaVersion := "2.12.2"

//libraryDependencies += "edu.stanford.nlp" % "stanford-corenlp" % "3.9.1"

//libraryDependencies += "edu.stanford.nlp" % "stanford-corenlp" % "3.9.2" classifier "models"



libraryDependencies ++= Seq(
    "edu.stanford.nlp" % "stanford-corenlp" % "3.9.1",
    "edu.stanford.nlp" % "stanford-corenlp" % "3.9.1" classifier "models"
)
