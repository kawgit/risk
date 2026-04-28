import edu.bu.pas.risk.territory.Territory;
import java.util.Arrays;
public class test {
    public static void main(String[] args) throws Exception {
        for (Territory t : Territory.values()) {
            System.out.println(t.name() + " -> " + t.getAdjacentTerritories().size());
        }
    }
}
