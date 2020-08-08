compile:
	javac -cp . com/perceptronnetwork/Setup.java

run: compile
	java com.perceptronnetwork.Setup

clean:
	find com -type f -name "*.class" -delete
