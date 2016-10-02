package com.jesh;

/**
 * Created by Jesh on 6/28/16.
 */
public class Main {
    public static void main(String[] args) {
        Runnable r = () -> System.out.println("Hello world!");
        r.run();
    }
}
