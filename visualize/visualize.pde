import de.voidplus.leapmotion.*;
import processing.net.*; 
Client myClient;
import g4p_controls.*;

//GUI STUFF----------------------//
GPanel panel;
GCheckbox[] buttons = new GCheckbox[6];

LeapMotion leap;
color colors[] = new color[6];
ArrayList<ArrayList<PVector>> trajectory = new ArrayList<ArrayList<PVector>>();

void initializeTrajectories(){
  trajectory.clear();
  for(int i = 0; i < 6; i++){
    trajectory.add(new ArrayList<PVector>());
  }
}

void setup() {
  size(800, 500);
  background(255);
  // ...
  myClient = new Client(this, "127.0.0.1", 50007); 
  leap = new LeapMotion(this);
  initializeTrajectories();
  for(int i = 0; i < colors.length; i++){
    colors[i] = color(random(255),random(255), random(255), 200);  
  }
  
  //GUI-----
  G4P.setGlobalColorScheme(GCScheme.RED_SCHEME);
  panel = new GPanel(this, 10, 10, 100, 400, "Options");
  String names[] = {"Palm", "Thumb", "Index", "Middle", "Ring", "Pinky"}; 
  for(int i = 0; i < buttons.length; i++){
      buttons[i] = new GCheckbox(this, 20, 40 + i*400/6, 90, 18, names[i]);
      buttons[i].setSelected(true);
      panel.addControl(buttons[i]);
  }
  frameRate(100);
}


String last_gesture = null;
void draw() {
  textSize(12);
  strokeWeight(10);
  String receive = myClient.readString();
  if(receive != null){
    println("Recibi : " + receive);    
    last_gesture = receive;
  }
  background(255);
  int fps = leap.getFrameRate();
  
  Hand right = leap.getRightHand();    
  Hand left = leap.getLeftHand();
  Hand dominant = null;
  if(right != null){
    dominant = right;
  }else if(left != null){
    dominant = left;
  }
  if(dominant != null){
    segment(dominant.getThumb().getVelocity(), 
            dominant.getIndexFinger().getVelocity(),
            dominant.getMiddleFinger().getVelocity(),
            dominant.getRingFinger().getVelocity(),
            dominant.getPinkyFinger().getVelocity()
            );
  }else{
    time = millis();
  }
  if(start){
    background(255,0,0);
    if(dominant != null){
      generateFrame(dominant.getPosition(), 
                    dominant.getThumb().getPositionOfJointTip(), 
                    dominant.getIndexFinger().getPositionOfJointTip(),
                    dominant.getMiddleFinger().getPositionOfJointTip(),
                    dominant.getRingFinger().getPositionOfJointTip(),
                    dominant.getPinkyFinger().getPositionOfJointTip()
                    );
      
      trajectory.get(0).add(dominant.getPosition());
      trajectory.get(1).add(dominant.getThumb().getPosition());
      trajectory.get(2).add(dominant.getIndexFinger().getPosition());
      trajectory.get(3).add(dominant.getMiddleFinger().getPosition());
      trajectory.get(4).add(dominant.getRingFinger().getPosition());
      trajectory.get(5).add(dominant.getPinkyFinger().getPosition());
    }
  }
  
  for (Hand hand : leap.getHands ()) {
    hand.draw();
  }
  
  //draw last recognized trajectory
  int p = 0;
  for(ArrayList<PVector> t : trajectory){
    if(buttons[p].isSelected()){
      for(PVector v : t){
        drawPoint(v, colors[p]);
      }
    }
    p++;
  }
  textMode(SHAPE);
  textSize(32);
  text("Ultimo gesto recibido : " + last_gesture, width/2 - 200, 30);     
}


ArrayList<String> info = new ArrayList<String>();
int num = 2000;
void generateHeader(Device device){
  info.add("Device" + device.toString());
  //sort info
  info.add("h_PosX h_PosY h_PosZ");  
  println(info.get(0));
  println(info.get(1));
}

void generateFrame(PVector p0, PVector p1, PVector p2, PVector p3, PVector p4, PVector p5){
  String str = "" + p0.x + " "  + p0.y + " " + p0.z + " ";
  str += "TYPE_THUMB # # # # "  + p1.x + " " + p1.y + " " + p1.z + " ";
  str += "TYPE_INDEX # # # # "  + p2.x + " " + p2.y + " " + p2.z + " ";
  str += "TYPE_MIDDLE # # # # " + p3.x + " " + p3.y + " " + p3.z + " ";
  str += "TYPE_RING # # # # "   + p4.x + " " + p4.y + " " + p4.z + " ";
  str += "TYPE_PINKY # # # # "  + p5.x + " " + p5.y + " " + p5.z + " ";
  println(str);
  info.add(str);
}

void drawPoint(PVector v, color c){
  pushStyle();
  fill(c);
  pushMatrix();
  stroke(c);
  //translate(v.x,v.y,v.z);  
  point(v.x,v.y);
  //sphere(1);
  popStyle();
  popMatrix();

}

boolean start = false;
boolean move = false;
float time = 0;
float MIN_SPEED = 650;
int MAX_TIME = 1000; 

void segment(PVector p1, PVector p2, PVector p3, PVector p4, PVector p5){  
  if(getSpeed(p1, p2, p3, p4, p5) <= MIN_SPEED){
    if(millis() - time > MAX_TIME && move){
      start = !start;
      move = false;      
      if(start){
        info.clear();
        initializeTrajectories();      
        generateHeader(leap.getDevices().get(0));
      }else{
        String[] data = new String[info.size()];
        data = info.toArray(data);                  
        saveStrings("gesture_" + num + ".txt", data);
        println("end gesture");
        println("Send to python");
        myClient.write("gesture_" + num + ".txt");    
        num++;
      }
    }
  }else{
    time = millis();  
    move = true;
  }
}

float getSpeed(PVector p1, PVector p2, PVector p3, PVector p4, PVector p5){
  PVector sum = new PVector();
  sum.add(p1);
  sum.add(p2);
  sum.add(p3);
  sum.add(p4);
  sum.add(p5);
  return sum.mag()/5.0;
}


//  myClient.write("Paging Python!"); // send whatever you need to send here
//  String receive = myClient.readString();
//  println("Recibi : " + receive);